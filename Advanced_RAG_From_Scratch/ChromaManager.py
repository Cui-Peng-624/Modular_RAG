import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv # type: ignore
from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain_chroma import Chroma # type: ignore
from langchain.document_loaders import PyPDFLoader # type: ignore 
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.docstore.document import Document # type: ignore
from uuid import uuid4

class ChromaVectorStoreManager:
    def __init__(self, collection_name: str = "default", persist_directory: str = None) -> None:
        # 设置代理
        os.environ["http_proxy"] = "127.0.0.1:7890"
        os.environ["https_proxy"] = "127.0.0.1:7890"

        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('ZETATECHS_API_KEY')
        base_url = os.getenv('ZETATECHS_API_BASE')

        # 初始化embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key, base_url=base_url)

        # 如果没有提供persist_directory，则使用默认路径
        if persist_directory is None:
            current_directory = os.getcwd()
            persist_directory = os.path.join(current_directory, '..', 'ChromaVDB')

        # 初始化Chroma向量存储
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        self.vector_store.add_documents(documents=documents_chunks, ids=uuids)

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        results = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
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
        """返回vector_store以便在chain中使用"""
        return self.vector_store