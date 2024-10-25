import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

import time
from uuid import uuid4
# import pinecone
from pinecone import Pinecone, ServerlessSpec
# from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import ServerlessSpec

from typing import Tuple, List

from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader # 处理单个PDF
from langchain.document_loaders import TextLoader # 处理单个TXT
# from langchain.document_loaders import PyPDFDirectoryLoader # 处理文件夹下的所有PDF
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

class PineconeManager:
    def __init__(self, index_name: str = "advanced-rag"):
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name # 默认"advanced-rag"
        self.ZETATECHS_API_KEY = os.getenv("ZETATECHS_API_KEY")
        self.ZETATECHS_API_BASE = os.getenv("ZETATECHS_API_BASE")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=self.ZETATECHS_API_KEY, base_url=self.ZETATECHS_API_BASE) # 1.使用第三方低价API；2.使用text-embedding-3-large适应3072的维度
        
        # Initialize Pinecone
        pc = Pinecone(api_key=self.PINECONE_API_KEY)
        
        # Create a serverless index if the index is not exist - 会报错：AttributeError: 'Pinecone' object has no attribute 'has_index' - 不应该啊
        # if not pc.has_index(self.index_name):
        #     pc.create_index(
        #         name=self.index_name, 
        #         dimension = 3072,
        #         metric = "cosine",
        #         spec = ServerlessSpec(
        #             cloud="aws", 
        #             region="us-east-1"
        #         )
        #     )
        # github上的版本
        # Create index if it doesn't exist
        if index_name not in [idx.name for idx in pc.list_indexes()]:
            spec = ServerlessSpec(cloud='aws', region='us-east-1')
            pc.create_index(name=index_name, dimension=3072, metric="cosine", spec=spec)

        # Wait for the index to be ready
        while not pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)

        self.index = pc.Index(self.index_name)

    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, namespace: str = "default") -> None: # 方法的返回类型注解为 None，表示该方法不返回任何值。返回类型注解使用箭头符号 ->，后跟 None。
        loader = PyPDFLoader(file_path) # 创建 PyPDFLoader 实例
        documents = loader.load() # 加载 PDF 文件并转换为文本数据
        splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap  = chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        print("1")

        vectorstore = PineconeVectorStore(index=self.index, namespace=namespace, embedding=self.embeddings)
        print("2")
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        vectorstore.add_documents(documents=documents_chunks, ids=uuids)

        print(f"Successfully uploaded {len(documents_chunks)} chunks from PDF file.")

    def upload_txt_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, namespace: str = "default") -> None:
        # 加载TXT文件并分块
        loader = TextLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)

        # 将文档块上传到 Pinecone
        vectorstore = PineconeVectorStore(index=self.index, namespace=namespace, embedding=self.embeddings)
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        vectorstore.add_documents(documents=documents_chunks, ids=uuids)

        print(f"Successfully uploaded {len(documents_chunks)} chunks from TXT file.")

    def upload_folder_files(self, folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, namespace: str = "default") -> None: # 加载一个文件夹下的所有文件，包括pdf和txt
        # 加载文件夹中的TXT文件
        txt_loader = DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader)
        txt_documents = txt_loader.load()

        # 加载文件夹中的PDF文件
        pdf_loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
        pdf_documents = pdf_loader.load()

        # 合并所有文件内容
        all_documents = txt_documents + pdf_documents

        # 分块
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(all_documents)

        # 将文档块上传到 Pinecone
        vectorstore = PineconeVectorStore(index=self.index, namespace=namespace, embedding=self.embeddings)
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        vectorstore.add_documents(documents=documents_chunks, ids=uuids)

        print(f"Successfully uploaded {len(documents_chunks)} chunks from folder containing PDF and TXT files.")

    def retrieval(self, query: str, namespace: str = "default", top_k: int = 3) -> Tuple[List, List]:
        vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings, namespace=namespace)
        results_with_metadata = vector_store.similarity_search_with_score(query, k=top_k)
        # results  = vector_store.similarity_search(query)
        results_only_str = [res.page_content for res, _ in results_with_metadata]
        return results_with_metadata, results_only_str