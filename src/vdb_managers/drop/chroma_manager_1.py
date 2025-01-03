import os
# 设置代理
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

from typing import List, Dict, Any
from dotenv import load_dotenv # type: ignore
from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain_chroma import Chroma # type: ignore
from uuid import uuid4
from rank_bm25 import BM25Okapi # type: ignore # BM25Okapi 是标准的BM25算法，rank_bm25 这个库中也有一些高级算法，例如：BM25L，BM25Plus
import numpy as np # type: ignore

# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\vdb_managers\chroma_manager.py\..\..
sys.path.append(project_root)

from document_processors.loader import DocumentLoader
from document_processors.splitter import DocumentSplitter
from sparse_retrievers.bm25_manager import BM25Manager
from model_utils.model_manage import ModelManage # ModelManage 中有 extract_metadata 方法

class ChromaManager:
    def __init__(self, embedded_model: str = "text-embedding-3-large", collection_name: str = "default", persist_directory: str = None) -> None:
        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('ZETATECHS_API_KEY')
        base_url = os.getenv('ZETATECHS_API_BASE')

        # 初始化embeddings
        self.embeddings = OpenAIEmbeddings(model=embedded_model, api_key=api_key, base_url=base_url)

        # 如果没有提供persist_directory，则使用默认路径
        if persist_directory is None:
            current_directory = os.getcwd()
            persist_directory = os.path.join(current_directory, 'ChromaVDB') # 在哪个文件中运行，就在该文件夹下建立chroma数据库

        # 初始化Chroma向量存储
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        
        # 初始化文档切分器
        self.splitter = DocumentSplitter()

        # 初始化BM25索引和documents列表
        self.bm25 = None
        self.documents = []
        
        # 初始化sparse_retriever
        self.sparse_retriever = BM25Manager()
        
        # 从现有数据库加载文档内容并初始化索引
        self._initialize_indices()

    def _initialize_indices(self) -> None:
        """从Chroma数据库初始化BM25索引和sparse_retriever"""
        try:
            # 获取所有文档
            all_documents = self.vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'uris', 'data', 'metadatas', 'included'])
            if all_documents and len(all_documents['documents']) > 0: # 如果文档存在，并且文档数量大于0
                # 更新documents列表
                self.documents = all_documents['documents']
                # 创建BM25索引
                self._create_bm25_index(self.documents)
                # 更新sparse_retriever
                self.sparse_retriever.add_documents(self.documents)
        except Exception as e:
            print(f"初始化索引时出错: {str(e)}")

    def _create_bm25_index(self, documents: List[str]) -> None: # 用法请参考：tests\rank_bm25_test.ipynb
        """创建BM25索引"""
        # 对文档进行分词
        tokenized_documents = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_documents)

    def get_vector_store(self):
        """返回vector_store实例
        
        Returns:
            Chroma向量存储实例
        """
        return self.vector_store # 返回langchain的chroma实例
        
    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        """上传PDF文件到向量数据库
        
        Args:
            file_path: PDF文件路径
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
        """
        # 使用DocumentLoader加载PDF
        documents = DocumentLoader.load_pdf(file_path)
        
        # 使用DocumentSplitter切分文档
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        
        # 生成唯一标识符并添加到向量存储
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        self.vector_store.add_documents(documents=documents_chunks, ids=uuids)
        
        # 更新BM25索引
        documents_content = [doc.page_content for doc in documents_chunks]
        self.documents.extend(documents_content)
        self._create_bm25_index(self.documents)
        
        # 更新sparse_retriever
        self.sparse_retriever.add_documents(documents_content)
    
    def upload_txt_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        """上传TXT文件到向量数据库
        
        Args:
            file_path: TXT文件路径
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
        """
        # 使用DocumentLoader加载TXT
        documents = DocumentLoader.load_txt(file_path)
        
        # 使用DocumentSplitter切分文档
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        
        # 生成唯一标识符并添加到向量存储
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        self.vector_store.add_documents(documents=documents_chunks, ids=uuids)
        
        # 更新BM25索引
        documents_content = [doc.page_content for doc in documents_chunks]
        self.documents.extend(documents_content)
        self._create_bm25_index(self.documents)
        
        # 更新sparse_retriever
        self.sparse_retriever.add_documents(documents_content)
    
    def upload_directory(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        """上传目录中的所有PDF和TXT文件到向量数据库
        
        Args:
            directory_path: 目录路径
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
        """
        # 使用DocumentLoader加载目录中的所有文件
        documents = DocumentLoader.load_directory(directory_path)
        
        # 使用DocumentSplitter切分文档
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        
        # 生成唯一标识符并添加到向量存储
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        self.vector_store.add_documents(documents=documents_chunks, ids=uuids)
        
        # 更新BM25索引
        documents_content = [doc.page_content for doc in documents_chunks]
        self.documents.extend(documents_content)
        self._create_bm25_index(self.documents)
        
        # 更新sparse_retriever
        self.sparse_retriever.add_documents(documents_content)

    # def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
    #     """执行相似性搜索
        
    #     Args:
    #         query: 查询文本
    #         k: 返回的结果数量
            
    #     Returns:
    #         包含相似文档内容和元数据的字典列表
    #     """
    #     results = self.vector_store.similarity_search(query, k=k)
    #     return [
    #         {
    #             "content": doc.page_content,
    #             "metadata": doc.metadata
    #         }
    #         for doc in results
    #     ]

    def dense_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
    # def similarity_search_with_score(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
    
        """执行带评分的相似性搜索
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            
        Returns:
            包含相似文档内容、元数据和相似分的字典列表
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
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
    def search(self, query: str, mode: str = "hybrid", k: int = 3, **kwargs) -> list[str]:
        """
        根据指定的搜索模式获取检索结果并返回内容列表
        
        Args:
            query: 查询文本
            mode: 搜索模式，可选值为 "hybrid", "dense"
            k: 返回的结果数量
            **kwargs: 其他参数，比如hybrid_search的dense_weight
            
        Returns:
            list[str]: 检索到的文档内容列表
            
        Raises:
            ValueError: 当指定的mode不支持时抛出异常
        """
        # 验证搜索模式
        valid_modes = ["hybrid", "dense"]
        if mode not in valid_modes:
            raise ValueError(f"不支持的搜索模式: {mode}。支持的模式包括: {', '.join(valid_modes)}")
        
        # 根据不同模式获取搜索结果
        if mode == "hybrid":
            dense_weight = kwargs.get("dense_weight", 0.5)
            results = self.hybrid_search(query, k=k, dense_weight=dense_weight)
            contents = [result["content"] for result in results]
            
        elif mode == "dense":
            results = self.dense_search(query, k=k)
            contents = [result["content"] for result in results]
            
        # else:  # similarity_with_score
        #     results = self.similarity_search_with_score(query, k=k)
        #     contents = [result["content"] for result in results]
        
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
            formatted_chunk = f"[文档片段 {i}]\n{cleaned_content}"
            formatted_chunks.append(formatted_chunk)
        
        # 使用分隔线连接所有文档片段
        formatted_context = "\n\n---\n\n".join(formatted_chunks)
        
        return formatted_context
