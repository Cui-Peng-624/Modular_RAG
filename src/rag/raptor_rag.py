from typing import List, Dict, Any, Optional, Literal
from .base_rag import BaseRAG
from src.vdb_managers.chroma_manager import ChromaManager_RAPTOR

class RaptorRAG(BaseRAG):
    """RAPTOR实现的RAG系统"""
    
    def __init__(self, model_manager: Optional[ModelManage] = None):
        # 使用RAPTOR版本的向量库管理器
        super().__init__(
            vdb_manager=ChromaManager_RAPTOR(),
            model_manager=model_manager
        )

    def retrieve(self, 
                query: str, 
                collection_name: str,
                search_direction: Literal["top_down", "bottom_up"] = "top_down",
                max_level: Optional[int] = None,
                include_children: bool = True,
                include_parents: bool = True,
                **kwargs) -> List[Dict[str, Any]]:
        """RAPTOR的层次化检索"""
        return self.vdb_manager.dense_search(
            collection_name=collection_name,
            query=query,
            search_direction=search_direction,
            max_level=max_level,
            include_children=include_children,
            include_parents=include_parents,
            **kwargs
        )

    def generate(self, 
                query: str, 
                context: List[Dict[str, Any]], 
                use_hierarchy: bool = True,
                **kwargs) -> str:
        """考虑层次结构的答案生成"""
        if use_hierarchy:
            # 根据文档的层级组织上下文
            organized_context = self._organize_hierarchical_context(context)
            return self.model_manager.generate_with_context(
                user_query=query,
                context_chunks=organized_context,
                **kwargs
            )
        return super().generate(query, context, **kwargs)

    def _organize_hierarchical_context(self, context: List[Dict[str, Any]]) -> List[str]:
        """将检索到的文档按层级组织"""
        # 按level排序
        sorted_docs = sorted(context, key=lambda x: x["metadata"]["level"])
        
        organized_texts = []
        for doc in sorted_docs:
            level = doc["metadata"]["level"]
            is_summary = doc["metadata"]["is_summary"]
            prefix = f"[Level {level} {'Summary' if is_summary else 'Detail'}]"
            organized_texts.append(f"{prefix}\n{doc['content']}")
            
            # 添加相关文档
            if "related_docs" in doc:
                for related in doc["related_docs"]:
                    rel_level = related["metadata"]["level"]
                    rel_is_summary = related["metadata"]["is_summary"]
                    rel_prefix = f"[Level {rel_level} {'Summary' if rel_is_summary else 'Detail'} - {related['relation'].capitalize()}]"
                    organized_texts.append(f"{rel_prefix}\n{related['content']}")
        
        return organized_texts 