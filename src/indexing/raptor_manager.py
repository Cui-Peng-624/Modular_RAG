from typing import List, Dict, Any
from uuid import uuid4
import numpy as np # type: ignore
from indexing.clustering import ClusteringFactory, ClusteringResult
from indexing.raptor_config import RaptorConfig
from indexing.extract_summaries_sync import extract_summaries_sync

import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent.absolute()) # src\indexing\raptor_manager.py
sys.path.append(project_root)
from model_utils.extract_embeddings_sync import extract_embeddings_sync

class RaptorManager:
    def __init__(self, config: RaptorConfig):
        """
        初始化RAPTOR管理器
        Args:
            config: RAPTOR配置对象
        """
        self.config = config
        self.clusterer = ClusteringFactory.get_clusterer(config) # 类似：GMMClusterer(threshold=self.gmm_threshold)
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        获取文本的嵌入向量
        Args:
            texts: 文本列表
        Returns:
            np.ndarray: 嵌入向量数组
        """
        try:
            # 使用同步版本的嵌入函数
            embeddings = extract_embeddings_sync(texts) # list
            # print(type(embeddings[0]), len(embeddings), len(embeddings[0])) # <class 'numpy.ndarray'> 18 3072
            return np.array(embeddings)
        except Exception as e:
            raise Exception(f"Failed to get embeddings: {str(e)}")
        
    def build_hierarchy(self, chunks: List[str]) -> Dict[str, Any]:
        """
        构建文档层次结构
        Args:
            chunks: 文档片段列表
        Returns:
            Dict包含每个层级的信息
        """
        if not self.config.enabled:
            return {}
            
        # 获取初始文档的嵌入向量
        current_embeddings = self._get_embeddings(chunks)

        levels = {}
        current_docs = chunks
        
        # 为初始文档生成ID，并在整个过程中保持跟踪
        doc_id_map = {str(uuid4()): doc for doc in chunks}  # 保存所有文档的ID映射。doc_id_map是一个全局的信息，包含在RAPTOR过程中所有id和chunk的对应
        current_doc_ids = list(doc_id_map.keys())  # 当前层级使用的文档ID
        
        for level in range(self.config.n_levels):
            # 执行聚类
            cluster_results: ClusteringResult = self.clusterer.fit_predict(current_embeddings)
            
            # 按聚类组织文档
            clusters: Dict[int, List[tuple]] = {}
            for doc_id, doc, label in zip(current_doc_ids, current_docs, cluster_results.labels):
                labels = [label] if isinstance(label, (int, np.integer)) else label
                for l in labels:
                    if l not in clusters:
                        clusters[l] = []
                    clusters[l].append((doc_id, doc))

            # 为每个聚类生成摘要
            summaries = []
            summary_ids = []
            parent_child_map = {}
            
            for label, docs in clusters.items():
                if label == -1 or len(docs) < self.config.min_cluster_size:
                    continue
                    
                cluster_doc_ids = [doc_id for doc_id, _ in docs]
                cluster_docs = [doc for _, doc in docs]
                
                try:
                    # 生成摘要
                    summary = extract_summaries_sync([" ".join(cluster_docs)])[0]
                    
                    # 为摘要生成唯一ID并保存到映射中
                    summary_id = str(uuid4())
                    doc_id_map[summary_id] = summary  # 将新生成的摘要添加到总映射中
                    
                    summaries.append(summary)
                    summary_ids.append(summary_id)
                    
                    # 记录父子关系
                    parent_child_map[summary_id] = cluster_doc_ids
                except Exception as e:
                    print(f"Warning: Failed to generate summary for cluster {label}: {str(e)}")
                    continue
            
            # 如果没有生成任何有效的摘要，提前结束
            if not summaries:
                break
                
            # 保存当前层级信息
            levels[level] = {
                "docs": current_docs,
                "doc_ids": current_doc_ids,  # 使用当前层级的文档ID
                "summaries": summaries,
                "summary_ids": summary_ids,
                "parent_child_map": parent_child_map,
                "cluster_centroids": cluster_results.centroids,
                "cluster_probabilities": cluster_results.probabilities,
                "doc_id_map": doc_id_map  # 保存完整的ID映射关系
            }
            
            # 更新下一层的输入
            current_docs = summaries
            current_doc_ids = summary_ids  # 更新当前文档ID为摘要ID
            
            # 如果是最后一层，不需要计算新的嵌入向量
            if level < self.config.n_levels - 1:
                current_embeddings = self._get_embeddings(summaries)
        
        return levels