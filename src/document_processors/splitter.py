from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
# 除了这个chunking方法，还有其他一些chunking的方法可以进一步探索
from langchain.docstore.document import Document # type: ignore

class DocumentSplitter:
    """文档切片器类，用于将文档切分成小块"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """初始化文档切片器
        
        Args:
            chunk_size: 每个文档块的最大字符数
            chunk_overlap: 相邻文档块之间的重叠字符数
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表切分成更小的块
        
        Args:
            documents: 要切分的Document对象列表
            
        Returns:
            切分后的Document对象列表
        """
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[Document]: # 这个函数是针对用户用键盘输入的str
        """
        将单个文本字符串切分成多个文档块
        
        Args:
            text: 要切分的文本字符串
            
        Returns:
            切分后的Document对象列表
        """
        return self.splitter.create_documents([text])
