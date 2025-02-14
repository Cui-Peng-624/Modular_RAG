from typing import List, Dict, Any, Optional, Literal
from src.vdb_managers.chroma_manager import ChromaManager
from src.model_utils.model_manage import ModelManage
from src.query_transformations.multi_query import generate_queries_multi_query_with_structured_output
from src.query_transformations.decomposition import generate_queries_decomposition_with_structured_output
from src.query_transformations.HyDE import generate_hyde_document
from src.query_transformations.step_back import step_back_question
from src.query_transformations.abstractness_analyzer import abstractness_analyzer

class BaseRAG:
    """基础RAG类，提供标准的RAG流程实现"""
    
    def __init__(self, 
                 vdb_manager: Optional[ChromaManager] = None,
                 model_manager: Optional[ModelManage] = None):
        self.vdb_manager = vdb_manager or ChromaManager()
        self.model_manager = model_manager or ModelManage()

    def upload(self, file_path: str, collection_name: str, **kwargs) -> None:
        """上传文档"""
        self.vdb_manager.upload_pdf_file(
            file_path=file_path,
            collection_name=collection_name,
            **kwargs
        )

    def transform_query(self, 
                       query: str, 
                       strategy: Literal["multi_query", "decomposition", "hyde", "step_back", "auto", "none"] = "none",
                       **kwargs) -> List[str]:
        """查询改写
        
        Args:
            query: 原始查询
            strategy: 改写策略
                - multi_query: 生成多个相似查询
                - decomposition: 将查询分解为子查询
                - hyde: 生成假设文档
                - step_back: 生成更抽象的查询
                - auto: 自动选择策略
                - none: 不进行改写
            **kwargs: 传递给具体策略的参数
        
        Returns:
            List[str]: 改写后的查询列表
        """
        if strategy == "none":
            return [query]
            
        if strategy == "auto":
            # 使用abstractness_analyzer判断问题类型
            is_abstract = abstractness_analyzer(query)
            if is_abstract:
                # 对于抽象问题，使用decomposition
                return generate_queries_decomposition_with_structured_output(query, **kwargs)
            else:
                # 对于具体问题，使用multi_query
                return generate_queries_multi_query_with_structured_output(query, **kwargs)
        
        # 使用指定的策略
        strategy_map = {
            "multi_query": generate_queries_multi_query_with_structured_output,
            "decomposition": generate_queries_decomposition_with_structured_output,
            "hyde": lambda q, **kw: [generate_hyde_document(q, **kw)],
            "step_back": lambda q, **kw: [step_back_question(q, **kw)]
        }
        
        return strategy_map[strategy](query, **kwargs)

    def retrieve(self, 
                queries: List[str], 
                collection_name: str, 
                merge_results: bool = True,
                **kwargs) -> List[Dict[str, Any]]:
        """检索相关文档
        
        Args:
            queries: 查询列表
            collection_name: 集合名称
            merge_results: 是否合并多个查询的结果
            **kwargs: 传递给检索方法的参数
        
        Returns:
            List[Dict[str, Any]]: 检索结果
        """
        all_results = []
        processed_ids = set()  # 用于去重
        
        for query in queries:
            results = self.vdb_manager.dense_search(
                collection_name=collection_name,
                query=query,
                **kwargs
            )
            
            if merge_results:
                # 去重并合并结果
                for result in results:
                    doc_id = result["metadata"].get("doc_id")
                    if doc_id and doc_id not in processed_ids:
                        all_results.append(result)
                        processed_ids.add(doc_id)
            else:
                all_results.extend(results)
        
        return all_results

    def generate(self, 
                query: str, 
                context: List[Dict[str, Any]], 
                **kwargs) -> str:
        """生成答案"""
        return self.model_manager.generate_with_context(
            user_query=query,
            context_chunks=[doc["content"] for doc in context],
            **kwargs
        )

    def query(self, 
              query: str, 
              collection_name: str, 
              query_transform_strategy: str = "none",
              **kwargs) -> str:
        """完整的RAG流程：查询改写+检索+生成
        
        Args:
            query: 用户查询
            collection_name: 集合名称
            query_transform_strategy: 查询改写策略
            **kwargs: 其他参数
        
        Returns:
            str: 生成的答案
        """
        # 1. 查询改写
        transformed_queries = self.transform_query(
            query, 
            strategy=query_transform_strategy,
            **kwargs.get("transform_kwargs", {})
        )
        
        # 2. 检索相关文档
        retrieved_docs = self.retrieve(
            transformed_queries, 
            collection_name,
            **kwargs.get("retrieve_kwargs", {})
        )
        
        # 3. 生成答案
        answer = self.generate(
            query, 
            retrieved_docs,
            **kwargs.get("generate_kwargs", {})
        )
        
        return answer 