# RAG 框架集成

## 一、项目概述
本项目致力于整合RAG不同阶段的各种方法，通过简洁的函数调用实现多种RAG方案。基础向量数据库使用[Chroma](https://github.com/chroma-core/chroma)。

---

## 二、核心功能

### 1. 文档上传阶段

``` python
def _upload_documents(
    self, 
    documents_type: Literal["pdf", "txt", "directory"],  # 支持的文件类型
    file_path: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 100, 
    auto_extract_metadata: bool = False, # 是否使用大模型自动异步提取每个chunk的特征作为元数据
    metadata: Dict[str, Any] = None, 
    collection_name: str = None, # 向量数据库的集合名称
    discription: str = None, # 向量数据库的描述 
    summarize_chunks: bool = False, # 是否使用大模型异步总结每个chunk，用于后续检索的时候使用
    similarity_metric: str = 'cosine' # 向量数据库的相似度度量方式
): 
    """统一文档上传入口"""
    pass

# PDF文件专用上传接口
def upload_pdf_file(*args, **kwargs) -> None:
    """上传PDF文件（继承_upload_documents参数）"""
    self._upload_documents("pdf", *args, **kwargs)    
```

### 2. 检索阶段

``` python
def search(
    self, 
    collection_name: str = None, # 向量数据库的集合名称
    query: str = None, # 用户的查询
    k: int = 3, # 检索的返回结果数量
    dense_weight: float = 0.5, # 密集检索的权重，取值范围为[0, 1]，如果为0，则只使用稀疏检索，如果为1，则只使用密集检索
    metadata_filter: Dict[str, Any] = None, # 元数据过滤
    fuzzy_filter: bool = False, # 是否启用模糊过滤（LLM自动匹配最高相似度的元数据）
    use_summary: bool = False, # 是否返回总结内容。True返回总结，False返回原文。
    colbert_rerank: bool = False, # 是否使用ColBERT重排序
    colbert_index_name: str = None, # ColBERT索引名称
    colbert_max_document_length: int = None, # ColBERT最大文档长度
    colbert_split_documents: bool = True, # 是否使用ColBERT分割文档
    colbert_k: int = 1, # ColBERT的k值
    **kwargs # 其他参数
) -> list[dict]:
    """混合检索核心方法"""
    pass
```

### 3. RAPTOR 集成
```python
class ChromaManager_RAPTOR:
    def __init__(
        self, 
        n_levels: int,  # 聚类层级数
        min_cluster_size: int  # 最小聚类尺寸
    ):
        """
        RAPTOR分层聚类实现
        - traversal模式：多层级遍历检索
        - collapsed模式：扁平化检索
        """
    
    def search(
        self,
        collection_name: str,
        query: str,
        k: int = 3,
        metadata_filter: Dict[str, Any] = None,
        search_type: Literal["traversal", "collapsed"] = None,
        d: int = 3,  # 遍历深度（traversal模式专用）
        top_k: int = 1  # 每层检索数（traversal模式专用）
    ) -> List[Dict[str, Any]]:
```

### 4. 高级查询功能
✅ [多查询扩展 (Multi Query)](https://github.com/Cui-Peng-624/Modular_RAG/blob/main/src/query_transformations/multi_query.py)  
✅ [问题分解 (Decomposition)](https://github.com/Cui-Peng-624/Modular_RAG/blob/main/src/query_transformations/decomposition.py) - 注意：其中单独回答和递归回答的代码仅仅是代码模板，需要根据实际情况进行修改。           
✅ [退步提问 (Step Back)](https://github.com/Cui-Peng-624/Modular_RAG/blob/main/src/query_transformations/step_back.py)  
✅ [假设文档嵌入 (HyDE)](https://github.com/Cui-Peng-624/Modular_RAG/blob/main/src/query_transformations/hyde.py)

---

## 三、Examples

以下是一个例子，我们希望在上传文件时自动提取元数据，总结每个chunk，并使用multi_query生成多个类似的问题，分别使用总结后的文档进行混合检索，最终生成的回答。

``` python
from src.query_transformations.multi_query import generate_queries_multi_query_with_structured_output
from src.vdb_managers.chroma_manager import ChromaManager
from src.generate.generate import generate_final_response

chroma_store = ChromaManager()
file_path = "***"
collection_name = "***"
chroma_store.upload_pdf_file(file_path = file_path, collection_name = collection_name, auto_extract_metadata = True, summarize_chunks = True)

query = "***"
queries = generate_queries_multi_query_with_structured_output(query)

total_retrieval_results = ""
for query in queries:
    # results = chroma_store.search(collection_name = collection_name, query = query, k = 3, dense_weight = 0.8, use_summary = True) # 也可以检索出来list[dict]，进行一定处理，再生成最终回答。
    results = chroma_store.get_formatted_context(collection_name = collection_name, query = query, k = 3, dense_weight = 0.8, use_summary = True)
    total_retrieval_results += results

final_answer = generate_final_response(query = query, retrieval_context = total_retrieval_results)
```
