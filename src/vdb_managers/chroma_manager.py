# 导入标准库
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv # type: ignore
import chromadb # type: ignore
import chromadb.utils.embedding_functions as embedding_functions # type: ignore
from datetime import datetime
from uuid import uuid4
import numpy as np # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore

# 设置代理
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

# 设置client
client = chromadb.PersistentClient(path = "ChromaVDB") # 注意，这里不是就在vdb_managers文件夹下创建永久化chroma数据库，而是这个py文件在哪里调用，就在哪里创建永久化chroma数据库，例如我再tests/chroma_manager_test.ipynb中调用，就在tests文件夹下创建永久化chroma数据库

# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\vdb_managers\chroma_manager.py\..\..
sys.path.append(project_root)

# 导入自定义的模块
from document_processors.loader import DocumentLoader
from document_processors.splitter import DocumentSplitter
from sparse_retrievers.bm25_manager import BM25Manager
from model_utils.AsyncApiClient import extract_metadata_sync

class ChromaManager:
    METADATA_REGISTRY_PATH = os.path.join(project_root, 'vdb_managers', 'metadata_registry.json')

    def __init__(self, embedded_model: str = "text-embedding-3-large") -> None:
        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('ZETATECHS_API_KEY')
        base_url = os.getenv('ZETATECHS_API_BASE')

        # 初始化embeddings
        self.openai_embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name=embedded_model
            )
        
        # 初始化文档切分器
        self.splitter = DocumentSplitter()
        
        # 初始化sparse_retriever
        self.sparse_retriever = BM25Manager()
        
        self._load_metadata_registry() # 每次初始化ChromaManager都会加载现有的全部数据到内存

    def _load_metadata_registry(self):
        """加载或初始化元数据注册表"""
        if os.path.exists(self.METADATA_REGISTRY_PATH):
            with open(self.METADATA_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                self.metadata_registry = json.load(f)
                # 确保所有值是集合类型
                for collection, metadata in self.metadata_registry.items(): # collection就是储存的向量知识库的名称
                    for key, values in metadata.items(): # key是metadata的key，value是一个list
                        self.metadata_registry[collection][key] = set(values)
        else:
            # 初始化为新的结构：每个 collection 有独立的 metadata 键值对
            self.metadata_registry = {}
            self._save_metadata_registry()

    def _save_metadata_registry(self):
        """保存元数据注册表到文件"""
        # 将集合转换为列表以便 JSON 序列化
        serializable_registry = {
            collection: {key: list(values) for key, values in metadata.items()}
            for collection, metadata in self.metadata_registry.items()
        }
        with open(self.METADATA_REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump(serializable_registry, f, ensure_ascii=False, indent=4)

    def _update_metadata_registry(self, collection_name: str, metadatas: List[Dict[str, Any]]): # metadatas类似：[{"category": str, "keywords": str}, {}, ...]
        """
        更新元数据注册表
        Args:
            collection_name: 集合名称
            metadatas: 上传文档的元数据列表
        """
        if collection_name not in self.metadata_registry:
            # 如果 collection 不存在，初始化为一个空字典
            self.metadata_registry[collection_name] = {}

        # 遍历每个 metadata，更新 registry
        for metadata in metadatas:
            for key, value in metadata.items():
                if key not in self.metadata_registry[collection_name]:
                    # 如果 key 不存在，初始化为一个空集合
                    self.metadata_registry[collection_name][key] = set()
                # 将值添加到集合中
                self.metadata_registry[collection_name][key].add(value)

        self._save_metadata_registry()

    def _get_vector_store(self, collection_name: str, discription: str = None, similarity_metric: str = "cosine") -> chromadb.Collection:
        """根据 collection_name 初始化或加载 Chroma 向量存储"""
        metadata = {
            "created_at": str(datetime.now())
        }
        if discription:  # 如果提供了 discription 参数，则添加到 metadata
            metadata["discription"] = discription

        if similarity_metric:
            metadata["hnsw:space"] = similarity_metric

        return client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.openai_embedding_function,
            metadata=metadata
        )
    
    # 这个好像废弃了
    # def _initialize_indices(self):
    #     """
    #     初始化文档和索引
    #     """
    #     # 假设文档存储在某个持久化存储中（如数据库或文件）
    #     # 这里需要根据实际情况加载文档
    #     if not hasattr(self, "vector_store"): # 检查实例中是否有vector_store属性
    #         raise ValueError("未找到向量存储，请先初始化或上传文档。")

    #     # 从 vector_store 中加载所有文档
    #     all_documents = self.vector_store.get_all_documents()  # 假设有此方法
    #     self.documents = [doc["content"] for doc in all_documents]

    #     # 初始化 BM25 索引
    #     self._create_bm25_index(self.documents)

    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, auto_extract_metadata: bool = True, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, similarity_metric: str = 'cosine') -> None:
        """
        上传PDF文件到向量数据库并提取元数据
        auto_extract_metadata：是否使用大模型自动提取每个chunk的特征作为元数据
        similarity_metric：l2, ip, consine
        """
        if collection_name is None:
            raise ValueError("你在上传文件的时候必须指定集合名称！")

        documents = DocumentLoader.load_pdf(file_path)
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)  # 1. list: [Document(metadata={'source': 'files/论文 - GraphRAG.pdf', 'page': 0}, page_content=), Document(), ...] # 2. 通过documents_chunks[i].page_content获取chunk内容
        
        metadatas = [] # [{"category": str, "keywords": str}, {}, ...]
        
        if auto_extract_metadata is True:  # 如果指定要求使用大模型提取元数据，不管有没有传入metadata，我们都不管
            extracted_metadata = extract_metadata_sync(file_path, documents_chunks)  # list: [{'category': '***', 'keywords': ['', '', '', '', '']}, {}, ...]
            for item in extracted_metadata: 
                category, keywords = item["category"], item["keywords"] # str, list[str]
                metadatas.append({"category": category, "keywords": keywords[0]}) # 只添加第一个关键词，因为chroma的metadata要求的value不能是list
        elif auto_extract_metadata is False and metadata is not None:  # 传入的metadata的格式也应该是dict
            metadatas = [metadata.copy() for _ in range(len(documents_chunks))]
        else: # auto_extract_metadata is False and metadata is None
            metadatas = [{"category": "default", "keywords": "default"} for _ in range(len(documents_chunks))]

        documents = [chunk.page_content for chunk in documents_chunks]
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]

        # 初始化对应的 vector_store
        vector_store = self._get_vector_store(collection_name=collection_name, discription=discription, similarity_metric = similarity_metric)

        # 更新元数据注册表 - 我们需要记录的是什么？是每个collection下面每个chunk的关键字，方便后续当我们需要使用关键字查询的时候进行匹配
        self._update_metadata_registry(collection_name, metadatas) # (str, list(dict))

        for metadata, documents_chunk in zip(metadatas, documents_chunks):
            metadata["source"] = documents_chunk.metadata.get("source", None)
            metadata["page"] = documents_chunk.metadata.get("page", None)

        vector_store.add(documents=documents, metadatas=metadatas, ids=uuids)

        # 更新 BM25 索引和文档内容
        self.sparse_retriever.add_documents(collection_name, documents)

    def upload_txt_file() -> None:
        """上传TXT文件到向量数据库并提取元数据"""
        pass

    def upload_directory() -> None:
        """上传目录中的所有PDF和TXT文件到向量数据库并提取元数据"""
        pass

    def dense_search(self, collection_name: str, query: str, k: int = 3, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        metadata_filter example:
            metadata_filter = {'author': {'$in': ['john', 'jill']}}
        """
        vector_store = self._get_vector_store(collection_name=collection_name)
        results = vector_store.query(
            query_texts = [query],
            where = metadata_filter,
            n_results = k,
        )
        return results
    
    def _get_metadata_from_vector_store(self, collection_name: str, document: str) -> Dict[str, Any]:
        """
        从向量数据库中获取指定文档的 metadata
        """
        vector_store = self._get_vector_store(collection_name=collection_name)
        results = vector_store.query(
            query_texts=[document],  # 使用文档内容作为查询
            n_results=1,  # 只需要返回一个匹配结果
            include=["metadatas"]  # 只需要 metadata # include参数，返回的是啥，这里只要求返回metadatas
        )
        if results and results.get("metadatas", [])[0]:
            return results["metadatas"][0][0]  # 返回第一个匹配的 metadata
        return {}

    def hybrid_search(self, collection_name: str, query: str, k: int = 3, dense_weight: float = 0.5, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """混合搜索方法，支持元数据过滤"""
        # 加载稀疏索引
        self.sparse_retriever.load_collection(collection_name)

        # 1. 获取密集向量搜索结果
        dense_results = self.dense_search(collection_name, query, k=k)
        dense_documents = dense_results.get('documents', [])[0]  # [[]]，[0]代表返回[]
        dense_metadatas = dense_results.get('metadatas', [])[0]
        dense_distances = dense_results.get('distances', [])[0]

        # 将密集搜索结果转换为字典
        dense_dict = {
            doc: {
                'score': 1 - distance,  # chroma的cosine计算是"1-余弦相似度"
                'metadata': metadata,
                'content': doc
            }
            for doc, metadata, distance in zip(dense_documents, dense_metadatas, dense_distances)
        }

        # 2. 获取稀疏搜索结果
        sparse_results = self.sparse_retriever.search(query, k=k)
        sparse_dict = {
            result['content']: {
                'score': result['score'],
                'content': result['content']
            }
            for result in sparse_results
        }

        # 3. 为稀疏搜索结果补充 metadata
        for doc in sparse_dict.keys():
            if doc not in dense_dict:  # 如果稀疏搜索的文档不在密集搜索结果中
                # 从向量数据库中查询 metadata
                metadata = self._get_metadata_from_vector_store(collection_name, doc)
                sparse_dict[doc]['metadata'] = metadata

        # 4. 融合结果
        all_docs = set(dense_dict.keys()) | set(sparse_dict.keys())
        hybrid_results = []

        for doc in all_docs:
            hybrid_score = 0.0
            metadata = {}

            if doc in dense_dict:
                hybrid_score += dense_weight * dense_dict[doc]['score']
                metadata = dense_dict[doc].get('metadata', {})

            if doc in sparse_dict:
                hybrid_score += (1 - dense_weight) * sparse_dict[doc]['score']
                # 如果 metadata 为空，尝试从稀疏搜索结果中获取
                if not metadata:
                    metadata = sparse_dict[doc].get('metadata', {})

            # 如果有元数据过滤条件，检查是否符合
            if metadata_filter:
                if not all(metadata.get(key) == value for key, value in metadata_filter.items()):
                    continue

            hybrid_results.append({
                'content': doc,
                'metadata': metadata,
                'score': hybrid_score,
            })

        # 按分数降序排序并返回前 k 个结果
        return sorted(hybrid_results, key=lambda x: x['score'], reverse=True)[:k]

    # 综合了上述的两种搜索模式，根据dense_weight的大小选择不同的搜索方式，返回list[str]
    def search(self, query: str, k: int = 3, metadata_filter: Dict[str, Any] = None, dense_weight: float = 0.5, collection_name: str = None, **kwargs) -> list[dict]:
        """
        根据指定的搜索模式和元数据过滤获取检索结果并返回内容列表

        Args:
            query: 查询文本
            k: 返回的结果数量
            metadata_filter: 元数据过滤条件，格式参考 Chroma 的 metadata filtering 文档
            dense_weight: 混合搜索模式中的密集搜索权重，默认值为 0.5，注意，此参数还可以用于表示单独的系数搜索和混合搜索
            collection_name: 集合名称
            **kwargs:

        Returns:
            list[dict]: 检索到的文档内容列表

        Raises:
            ValueError: 当指定的 mode 不支持时抛出异常
        """
        if collection_name:
            vector_store = self._get_vector_store(collection_name=collection_name)
        else:
            raise ValueError("您输入的集合名称错误！")
        
        # 检查dense_weight是否在0-1之间
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight的范围是0-1。0代表仅使用稀疏搜索，1代表仅使用密集搜索。")

        # 根据dense_weight选择不同的搜索模式
        results = self.hybrid_search(
            collection_name=collection_name,
            query=query,
            k=k,
            dense_weight=dense_weight,
            metadata_filter=metadata_filter
        )

        return results

    # 获取格式化的上下文字符串。此函数会先调用search获取内容列表，
    # 然后将其转换为格式化的字符串（就是拼接在一起）。
    def get_formatted_context(self, query: str, k: int = 3, metadata_filter: Dict[str, Any] = None, dense_weight: float = 0.5, collection_name: str = None, **kwargs) -> str:
        """
        获取格式化的上下文字符串。此函数会先调用search获取内容列表，
        然后将其转换为格式化的字符串。
        
        Args:
            query: 查询文本
            mode: 搜索模式，可选值为 "hybrid", "dense"
            k: 返回的结果数量
            metadata_filter: 元数据过滤条件
            dense_weight: 混合搜索模式中的密集搜索权重，默认值为 0.5
            collection_name: 集合名称
            **kwargs: 其他参数，比如 hybrid_search 的额外参数
            
        Returns:
            str: 格式化后的上下文字符串
        """
        # 获取内容列表
        results = self.search(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
            dense_weight=dense_weight,
            collection_name=collection_name,
            **kwargs
        )
        
        # 如果没有检索到内容，返回提示信息
        if not results:
            return "未检索到相关内容。"

        # 格式化每个文档片段
        formatted_chunks = []
        for i, result in enumerate(results, 1):
            # 清理文本（移除多余的空白字符）
            cleaned_content = " ".join(result["content"].split())
            # 添加编号和格式化
            formatted_chunk = f"[文档片段 {i}]\n{cleaned_content}"
            formatted_chunks.append(formatted_chunk)
        
        # 使用分隔线连接所有文档片段
        formatted_context = "\n\n---\n\n".join(formatted_chunks)
        
        return formatted_context

