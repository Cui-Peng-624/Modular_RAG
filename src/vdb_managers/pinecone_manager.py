import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

import time
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec # type: ignore
from typing import Tuple, List, Dict, Any

from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain_pinecone import PineconeVectorStore # type: ignore
from dotenv import load_dotenv # type: ignore

# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\tests\..\src
sys.path.append(project_root)

from document_processors.loader import DocumentLoader
from document_processors.splitter import DocumentSplitter
from sparse_retrievers.bm25_manager import BM25Manager

class PineconeManager:
    def __init__(self, index_name: str = "rag"):
        """初始化Pinecone管理器
        
        Args:
            index_name: Pinecone索引名称，默认为"advanced-rag"
        """
        # 加载环境变量
        load_dotenv()
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.ZETATECHS_API_KEY = os.getenv("ZETATECHS_API_KEY")
        self.ZETATECHS_API_BASE = os.getenv("ZETATECHS_API_BASE")
        
        # 初始化embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            api_key=self.ZETATECHS_API_KEY, 
            base_url=self.ZETATECHS_API_BASE
        )
        
        # 初始化Pinecone
        pc = Pinecone(api_key=self.PINECONE_API_KEY)
        
        # 如果索引不存在则创建
        if index_name not in [idx.name for idx in pc.list_indexes()]:
            spec = ServerlessSpec(cloud='aws', region='us-east-1')
            pc.create_index(
                name=index_name, 
                dimension=3072, 
                metric="cosine", 
                spec=spec
            )

        # 等待索引就绪
        while not pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)

        self.index = pc.Index(self.index_name)
        
        self.sparse_retriever = BM25Manager()
        
    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, namespace: str = "default") -> None:
        """上传PDF文件到向量数据库
        
        Args:
            file_path: PDF文件路径
            chunk_size: 文档块大���
            chunk_overlap: 文档块重叠大小
            namespace: Pinecone命名空间
        """
        # 使用DocumentLoader加载PDF
        documents = DocumentLoader.load_pdf(file_path)
        
        # 使用DocumentSplitter切分文档
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        
        # 生成唯一标识符并添加到向量存储
        vectorstore = PineconeVectorStore(index=self.index, namespace=namespace, embedding=self.embeddings)
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        vectorstore.add_documents(documents=documents_chunks, ids=uuids)
        
        print(f"成功上传 {len(documents_chunks)} 个文档块。")
        
        # 同时更新BM25索引
        documents_content = [doc.page_content for doc in documents_chunks]
        self.sparse_retriever.add_documents(documents_content)

    def upload_txt_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, namespace: str = "default") -> None:
        """上传TXT文件到向量数据库
        
        Args:
            file_path: TXT文件路径
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
            namespace: Pinecone命名空间
        """
        # 使用DocumentLoader加载TXT
        documents = DocumentLoader.load_txt(file_path)
        
        # 使用DocumentSplitter切分文档
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        
        # 生成唯一标识符并添加到向量存储
        vectorstore = PineconeVectorStore(index=self.index, namespace=namespace, embedding=self.embeddings)
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        vectorstore.add_documents(documents=documents_chunks, ids=uuids)
        
        print(f"成功上传 {len(documents_chunks)} 个文档块。")

    def upload_directory(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, namespace: str = "default") -> None:
        """上传目录中的所有PDF和TXT文件到向量数据库
        
        Args:
            directory_path: 目录路径
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
            namespace: Pinecone命名空间
        """
        # 使用DocumentLoader加载目录中的所有文件
        documents = DocumentLoader.load_directory(directory_path)
        
        # 使用DocumentSplitter切分文档
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        
        # 生成唯一标识符并添加到向量存储
        vectorstore = PineconeVectorStore(index=self.index, namespace=namespace, embedding=self.embeddings)
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        vectorstore.add_documents(documents=documents_chunks, ids=uuids)
        
        print(f"成功上传 {len(documents_chunks)} 个文档块。")

    def retrieval(self, query: str, namespace: str = "default", top_k: int = 3) -> Tuple[List[Dict[str, Any]], List[str]]:
        """执行相似性搜索
        
        Args:
            query: 查询文本
            namespace: Pinecone命名空间
            top_k: 返回的结果数量
            
        Returns:
            包含相似文档及其评分的组，以及仅包含文档内容的列表
        """
        vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings, namespace=namespace)
        results_with_metadata = vector_store.similarity_search_with_score(query, k=top_k)
        results_only_str = [res.page_content for res, _ in results_with_metadata]
        return results_with_metadata, results_only_str

    def hybrid_search(self, query: str, namespace: str = "default", k: int = 3, dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        """混合搜索方法
        
        Args:
            query: 查询文本
            namespace: Pinecone命名空间
            k: 返回结果数量
            dense_weight: 密集向量搜索的权重(0-1)
            
        Returns:
            混合搜索结果列表，包含文档内容、元数据和综合得分
        """
        # 1. 获取密集向量搜索结果
        dense_results, _ = self.retrieval(query, namespace=namespace, top_k=k)
        dense_dict = {
            result[0].page_content: {
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