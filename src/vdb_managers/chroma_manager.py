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
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\tests\..\src
sys.path.append(project_root)

from document_processors.loader import DocumentLoader
from document_processors.splitter import DocumentSplitter
from sparse_retrievers.bm25_manager import BM25Manager

class ChromaManager:
    def __init__(self, collection_name: str = "default", persist_directory: str = None) -> None:
        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('ZETATECHS_API_KEY')
        base_url = os.getenv('ZETATECHS_API_BASE')

        # 初始化embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key, base_url=base_url)

        # 如果没有提供persist_directory，则使用默认路径
        if persist_directory is None:
            current_directory = os.getcwd() # 对应测试文件的路径，而不是"chroma_manager.py"的路径
            # persist_directory = os.path.join(current_directory, '..', 'ChromaVDB')
            persist_directory = os.path.join(current_directory, 'ChromaVDB') # 在哪个文件中运行，就在该文件夹下建立chroma数据库

        # 初始化Chroma向量存储
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        
        # 初始化文档切分器
        self.splitter = DocumentSplitter()

        # 添加BM25索引
        self.bm25 = None
        self.documents = []  # 存储文档原文
        
        self.sparse_retriever = BM25Manager()
        
    def _create_bm25_index(self, documents: List[str]) -> None: # 用法请参考：tests\rank_bm25_test.ipynb
        """创建BM25索引"""
        # 对文档进行分词
        tokenized_documents = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_documents)
        
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
        self.documents.extend([doc.page_content for doc in documents_chunks])
        self._create_bm25_index(self.documents)
        
        # 同时更新BM25索引
        documents_content = [doc.page_content for doc in documents_chunks]
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

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """执行相似性搜索
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            
        Returns:
            包含相似文档内容和元数据的字典列表
        """
        results = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
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
    
    def get_vector_store(self):
        """返回vector_store实例
        
        Returns:
            Chroma向量存储实例
        """
        return self.vector_store # self使langchain的chroma，这里返回的是本身chroma的实例

    def hybrid_search(self, query: str, k: int = 3, dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        """混合搜索方法
        
        Args:
            query: 查询文本
            k: 返回结果数量
            dense_weight: 密集向量搜索的权重(0-1)
            
        Returns:
            混合搜索结果列表，包含文档内容、元数据和综合得分
        """
        # 1. 获取密集向量搜索结果
        dense_results = self.similarity_search_with_score(query, k=k)
        dense_dict = {
            result[0].page_content: { # result[0]
                'score': result[1],
                'metadata': result[0].metadata,
                'content': result[0].page_content
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
            
            # 如果文档在密集检索结果中
            if doc in dense_dict:
                hybrid_score += dense_weight * dense_dict[doc]['score']
                metadata = dense_dict[doc].get('metadata', {})
                
            # 如果文档在稀疏检索结果中
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