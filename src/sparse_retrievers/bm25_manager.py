import numpy as np # type: ignore
from rank_bm25 import BM25Okapi # type: ignore
from typing import List, Dict, Any

class BM25Manager:
    def __init__(self) -> None:
        """初始化BM25管理器"""
        self.bm25 = None
        self.documents: List[str] = []
        
    def add_documents(self, documents: List[str]) -> None:
        """添加文档到BM25索引
        
        Args:
            documents: 文档列表
        """
        self.documents.extend(documents)
        self._create_bm25_index(self.documents)
        
    def _create_bm25_index(self, documents: List[str]) -> None:
        """创建BM25索引
        
        Args:
            documents: 文档列表
        """
        tokenized_documents = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_documents)
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """执行BM25搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            搜索结果列表，包含文档内容和分数
        """
        if not self.bm25:
            raise ValueError("未创建BM25索引，请先添加文档")
            
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # 归一化分数
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # 获取top-k结果
        top_k_indices = np.argsort(normalized_scores)[-k:][::-1]
        
        return [{
            'content': self.documents[idx],
            'score': float(normalized_scores[idx])
        } for idx in top_k_indices]
