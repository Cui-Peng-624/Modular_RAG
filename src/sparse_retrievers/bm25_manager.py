import os
import json
import numpy as np # type: ignore
from rank_bm25 import BM25Okapi # type: ignore # BM25Okapi 是标准的BM25算法，rank_bm25 这个库中也有一些高级算法，例如：BM25L，BM25Plus
from typing import List, Dict, Any

from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\sparse_retrievers\bm25_manager.py\..\..
# __file__ 是 Python 中的一个特殊变量，表示当前脚本的文件路径

class BM25Manager:
    def __init__(self, storage_path: str = "bm25_collections.json") -> None:
        """初始化BM25管理器"""
        self.bm25 = None
        self.documents: List[str] = []
        self.storage_path = os.path.join(project_root, 'sparse_retrievers', storage_path)  # 本地持久化文件路径 - 在sparse_retrievers文件夹下创建bm25_collections.json文件
        self.collections = self._load_collections()

    def _load_collections(self) -> Dict[str, List[str]]:
        """加载本地持久化的文档集合"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_collections(self) -> None:
        """保存文档集合到本地文件"""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.collections, f, ensure_ascii=False, indent=4)

    def add_documents(self, collection_name: str, documents: List[str]) -> None: # documents就是chunk中的page_content组成的list
        """添加文档到BM25索引并持久化
        
        Args:
            collection_name: 集合名称
            documents: 文档列表
        """
        # 更新本地持久化文件
        if collection_name not in self.collections.keys():
            self.collections[collection_name] = []
        self.collections[collection_name].extend(documents)
        self.collections[collection_name] = list(set(self.collections[collection_name]))  # 去重。set是无序的，所以转会list可能会导致顺序改变，不用担心。
        self._save_collections()

        # 更新内存中的BM25索引 - 这里似乎没有必要，因为如果我们想调用稀疏搜索，必然会调用load_collection，load_collection中也会_create_bm25_index，但我们仍然不删除，因为更新一下也没坏处
        self._create_bm25_index(self.collections[collection_name])

    def _create_bm25_index(self, documents: List[str]) -> None:
        """创建BM25索引
        
        Args:
            documents: 文档列表
        """
        tokenized_documents = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_documents)
        self.documents = documents

    def load_collection(self, collection_name: str) -> None:
        """加载指定集合的文档并初始化BM25索引
        
        Args:
            collection_name: 集合名称
        """
        if collection_name not in self.collections.keys():
            raise ValueError(f"集合 {collection_name} 不存在，请先上传文档")
        self._create_bm25_index(self.collections[collection_name])

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """执行BM25搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            搜索结果列表，包含文档内容和分数
        """
        if not self.bm25:
            raise ValueError("未创建BM25索引，请先加载集合")
            
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query) # numpy.ndarray
        
        # 归一化分数
        normalized_scores = self.normalize_scores(scores)
        
        # 获取top-k结果
        # argsort：返回排序后的索引，索引按实际值的大小由小到大排列
        top_k_indices = np.argsort(normalized_scores)[-k:][::-1] # [-k:]：取最后k个。[::-1]：反转
        # 最后取出来的结果由大到小
        
        return [{
            'content': self.documents[idx],
            'score': float(normalized_scores[idx])
        } for idx in top_k_indices]

    def normalize_scores(self, scores):
        score_range = scores.max() - scores.min()
        if score_range == 0:
            return np.zeros_like(scores) if scores.min() == 0 else np.ones_like(scores)
        return (scores - scores.min()) / score_range
