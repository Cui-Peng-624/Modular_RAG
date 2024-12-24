import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader # type: ignore
from langchain.docstore.document import Document # type: ignore

class DocumentLoader:
    """文档加载器类，支持加载PDF、TXT文件以及包含这些文件的文件夹"""
    
    @staticmethod # 静态方法 - 静态方法既不依赖于类的实例（即不需要self参数），也不依赖于类本身（即不需要cls参数）。
    def load_pdf(file_path: str) -> List[Document]:
        """加载单个PDF文件
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            包含文档内容的Document对象列表
        """
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_txt(file_path: str) -> List[Document]:
        """加载单个TXT文件
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            包含文档内容的Document对象列表
        """
        loader = TextLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_directory(directory_path: str) -> List[Document]:
        """加载目录中的所有PDF和TXT文件
        
        Args:
            directory_path: 目录路径
            
        Returns:
            包含所有文档内容的Document对象列表
        """
        # 加载TXT文件
        txt_loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader)
        txt_documents = txt_loader.load()
        
        # 加载PDF文件
        pdf_loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
        pdf_documents = pdf_loader.load()
        
        # 合并所有文档
        return txt_documents + pdf_documents
