# 导入标准库
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv # type: ignore

##########################################################################################################################
# from langchain_openai import OpenAIEmbeddings # type: ignore
# from langchain_chroma import Chroma # type: ignore
import chromadb # type: ignore
import chromadb.utils.embedding_functions as embedding_functions # type: ignore
from datetime import datetime

client = chromadb.PersistentClient(path = "ChromaVDB") # 注意，这里不是就在vdb_managers文件夹下创建永久化chroma数据库，而是这个py文件在哪里调用，就在哪里创建永久化chroma数据库，例如我再tests/chroma_manager_test.ipynb中调用，就在tests文件夹下创建永久化chroma数据库
##########################################################################################################################

from uuid import uuid4
import numpy as np # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore

# 设置代理
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

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

        ###########################################################################################################################
        # 初始化embeddings
        self.openai_embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name=embedded_model
            )
        ###########################################################################################################################
        
        # 初始化文档切分器
        self.splitter = DocumentSplitter()

        # 初始化BM25索引和documents列表
        self.bm25 = None
        self.documents = []
        
        # 初始化sparse_retriever
        self.sparse_retriever = BM25Manager()
        
        self._load_metadata_registry()

    def _create_bm25_index(self, documents: List[str]) -> None:
        """
        创建BM25索引
        Args:
            documents: 文档内容列表
        """
        # 对文档进行分词
        tokenized_documents = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_documents)

    def _load_metadata_registry(self):
        """加载或初始化元数据注册表"""
        if os.path.exists(self.METADATA_REGISTRY_PATH):
            with open(self.METADATA_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                self.metadata_registry = json.load(f)
        else:
            # 初始化为新的结构：每个类别有独立的 keywords
            self.metadata_registry = {
                "categories": {}  # 每个类别对应一个关键词列表
            }
            self._save_metadata_registry()

    def _save_metadata_registry(self):
        """保存元数据注册表到文件"""
        with open(self.METADATA_REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_registry, f, ensure_ascii=False, indent=4)

    def _update_metadata_registry(self, categories: List[str], keywords: List[str]):
        """更新元数据注册表"""
        for category in categories:
            if category not in self.metadata_registry["categories"].keys():
                # 如果类别不存在，初始化为一个空列表
                self.metadata_registry["categories"][category] = []

            # 将新的关键词添加到对应类别的关键词列表中，避免重复
            existing_keywords = set(self.metadata_registry["categories"][category])
            new_keywords = set(keywords)
            self.metadata_registry["categories"][category] = list(existing_keywords.union(new_keywords))

        self._save_metadata_registry()

    def _get_vector_store(self, collection_name: str, similarity_metric: str, discription: str = None) -> chromadb.Collection:
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

    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, auto_extract_metadata: bool = True, metadata: Dict[str, Any] = None, collection_name: str = None, similarity_metric: str = 'cosine') -> None:
        """
        上传PDF文件到向量数据库并提取元数据
        auto_extract_metadata：是否使用大模型自动提取每个chunk的特征作为元数据
        similarity_metric：l2, ip, consine
        """
        documents = DocumentLoader.load_pdf(file_path)
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)  # 1. list: [Document(metadata={'source': 'files/论文 - GraphRAG.pdf', 'page': 0}, page_content=), Document(), ...] # 2. 通过documents_chunks[i].page_content获取chunk内容
        
        # 提取元数据
        if auto_extract_metadata and metadata is None:  # 如果指定要求使用大模型提取元数据，并且没有自己定义metadata
            extracted_metadata = extract_metadata_sync(file_path, documents_chunks)  # list: [{'category': '***', 'keywords': ['', '', '', '', '']}, {}, ...]
            categorized_chunks = {}  # {'计算机': {'chunks': [Document(metadata={'source': 'files/论文 - GraphRAG.pdf', 'page': 0}, page_content=), [], ...], 'keywords': {'', '', '', '', ''}}, "医学": {'chunks': [], 'keywords': set()}, ...}
            for chunk, meta in zip(documents_chunks, extracted_metadata):  # 将documents_chunks和extracted_metadata一一对应
                category = meta.get("category", "未分类")
                keywords = meta.get("keywords", [])
                if category not in categorized_chunks.keys():
                    categorized_chunks[category] = {"chunks": [], "keywords": set()}
                categorized_chunks[category]["chunks"].append(chunk)
                categorized_chunks[category]["keywords"].update(keywords)  # update：set({})的方法
        elif metadata is not None:  # 如果用户自定义了metadata
            # 将所有 chunks 归为一个默认类别
            categorized_chunks = {
                collection_name or "default": {  # 如果用户未提供 collection_name，则使用默认类别
                    "chunks": documents_chunks,
                    "keywords": set(metadata.get("keywords", []))  # 如果用户没有提供 keywords，则为空。有关python的set请参考：python_set.md
                }
            }
            # 删除 metadata 中的 keywords，避免后续重复
            if "keywords" in metadata:
                del metadata["keywords"]
            # print(categorized_chunks[collection_name]["keywords"], type(categorized_chunks[collection_name]["keywords"]))
        else:
            raise ValueError("auto_extract_metadata = True 时不能指定 metadata！")

        # 更新元数据注册表并按类别处理向量数据库
        for category, data in categorized_chunks.items():
            # 如果用户提供了 collection_name，则所有数据都上传到该 collection
            target_collection_name = collection_name or category # 优先使用collection_name，如果collection_name为空，0或False，再使用category

            # 更新元数据注册表
            self._update_metadata_registry([target_collection_name], list(data["keywords"]))

            # 初始化对应的 vector_store
            vector_store = self._get_vector_store(similarity_metric = similarity_metric, collection_name=target_collection_name)

            # 为每个 chunk 生成 UUID 并添加到向量数据库
            uuids = [str(uuid4()) for _ in range(len(data["chunks"]))]
            texts = [chunk.page_content for chunk in data["chunks"]]
            metadatas = [
                {
                    "source": chunk.metadata.get("source", None),
                    "keywords": ", ".join(data["keywords"]),  # 如果没有 keywords，则为空字符串
                    "page": chunk.metadata.get("page", None),  # 假设 chunk.metadata 包含页码信息
                    **(metadata or {})  # 合并用户自定义的 metadata
                }
                for chunk in data["chunks"]
            ]
            # print(metadatas)
            vector_store.add(documents=texts, metadatas=metadatas, ids=uuids)

        # 更新 BM25 索引和文档内容
        documents_content = [doc.page_content for doc in documents_chunks]
        self.documents.extend(documents_content)
        self._create_bm25_index(self.documents)
        self.sparse_retriever.add_documents(documents_content)

    def upload_txt_file() -> None:
        """上传TXT文件到向量数据库并提取元数据"""
        pass

    def upload_directory() -> None:
        """上传目录中的所有PDF和TXT文件到向量数据��并提取元数据"""
        pass

    def dense_search(self, collection_name: str, query: str, k: int = 3, similarity_metric: str = 'cosine') -> List[Dict[str, Any]]:
        vector_store = self._get_vector_store(collection_name=collection_name, similarity_metric = similarity_metric)
        results = vector_store.query(
            query_texts = [query],
            n_results = k,
        )
        return results
    
    def hybrid_search(self, query: str, k: int = 3, dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        """混合搜索方法"""
        if not self.documents:
            # 尝试重新初始化索引
            self._initialize_indices()
            if not self.documents:
                raise ValueError("未找到任何文档，请先添加文档到向量数据库")
        
        # 1. 获取密集向量搜索结果
        dense_results = self.similarity_search_with_score(query, k=k)
        dense_dict = {
            result['content']: {
                'score': result['score'],
                'metadata': result['metadata'],
                'content': result['content']
            }
            for result in dense_results
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
        
        # 3. 融合结果
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
                
            hybrid_results.append({
                'content': doc,
                'metadata': metadata,
                'score': hybrid_score,
                'dense_score': dense_dict.get(doc, {}).get('score', 0.0),
                'sparse_score': sparse_dict.get(doc, {}).get('score', 0.0)
            })
        
        # 4. 根据混合得分排序并返回top-k结果
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_results[:k]

    # 综合了上述的两种搜索模式，返回list[str]
    def search(self, query: str, mode: str = "hybrid", k: int = 3, metadata_filter: Dict[str, Any] = None, **kwargs) -> list[str]:
        """
        根据指定的搜索模式和元数据过滤获取检索结果并返回内容列表

        Args:
            query: 查询文本
            mode: 搜索模式，可选值为 "hybrid", "dense"
            k: 返回的结果数量
            metadata_filter: 元数据过滤条件，格式参考 Chroma 的 metadata filtering 文档
            **kwargs: 其他参数，比如 hybrid_search 的 dense_weight

        Returns:
            list[str]: 检索到的文档内容列表

        Raises:
            ValueError: 当指定的 mode 不支持时抛出异常
        """
        valid_modes = ["hybrid", "dense"]
        if mode not in valid_modes:
            raise ValueError(f"不支持的搜索模式: {mode}。支持的模式包括: {', '.join(valid_modes)}")

        # 判断是否需要根据元数据过滤搜索
        collection_name = None
        if metadata_filter:
            # 假设元数据过滤包含 'category' 字段
            collection_name = metadata_filter.get("category", None)

        if collection_name:
            vector_store = self._get_vector_store(collection_name=collection_name)
        else:
            vector_store = self._get_vector_store(collection_name="default")

        # 根据不同模式获取搜索结果
        if mode == "hybrid":
            dense_weight = kwargs.get("dense_weight", 0.5)
            results = self.hybrid_search(query, k=k, dense_weight=dense_weight)
            contents = [result["content"] for result in results]

        elif mode == "dense":
            results = self.dense_search(query, k=k)
            contents = [result["content"] for result in results]

        return contents

    # 获取格式化的上下文字符串。此函数会先调用search获取内容列表，
    # 然后将其转换为格式化的字符串（就是拼接在一起）。
    def get_formatted_context(self, query: str, mode: str = "hybrid", k: int = 3, **kwargs) -> str:
        """
        获取格式化的上下文字符串。此函数会先调用get_combined_contents获取内容列表，
        然后将其转换为格式化的字符串。
        
        Args:
            query: 查询文本
            mode: 搜索模式，可选值为 "hybrid", "similarity", "similarity_with_score"
            k: 返回的结果数量
            **kwargs: 其他参数，比如hybrid_search的dense_weight
            
        Returns:
            str: 格式化后的上下文字符串
        """
        # 获取内容列表
        contents = self.search(query, mode=mode, k=k, **kwargs)
        
        # 格式化每个文档片段
        formatted_chunks = []
        for i, content in enumerate(contents, 1):
            # 清理文本（移除多余的空白字符）
            cleaned_content = " ".join(content.split())
            # 添加编号和格式化
            formatted_chunk = f"[���档片段 {i}]\n{cleaned_content}"
            formatted_chunks.append(formatted_chunk)
        
        # 使用分隔线连接所有文档片段
        formatted_context = "\n\n---\n\n".join(formatted_chunks)
        
        return formatted_context