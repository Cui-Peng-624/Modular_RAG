import os
os.environ["OMP_NUM_THREADS"] = "1"

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np # type: ignore
from sklearn.mixture import GaussianMixture # type: ignore
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering # type: ignore
import umap # type: ignore
from dataclasses import dataclass
from indexing.raptor_config import RaptorConfig

@dataclass
class ClusteringResult:
    """聚类结果的标准化输出格式"""
    labels: np.ndarray  # 聚类标签
    n_clusters: int     # 聚类数量
    centroids: Optional[np.ndarray] = None  # 聚类中心（如果有）
    probabilities: Optional[np.ndarray] = None  # 概率分布（如果有）
    
class BaseClusterer(ABC):
    """
    聚类算法的基类
    ABC：抽象基类，用于定义抽象方法
    """
    @abstractmethod # 子类必须重新实现fit_predict方法
    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        """执行聚类并返回结果"""
        pass

class GMMClusterer(BaseClusterer):
    """基于高斯混合模型的聚类"""
    def __init__(self, threshold: float = 0.1, random_state: int = 42):
        self.threshold = threshold
        self.random_state = random_state
        
    def _get_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 50) -> int:
        """使用BIC准则确定最优聚类数"""
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=self.random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
            
        return n_clusters[np.argmin(bics)]
    
    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        n_clusters = self._get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        gm.fit(embeddings)
        
        # 获取概率分布
        probs = gm.predict_proba(embeddings)
        # 根据阈值确定标签
        labels = np.array([np.where(prob > self.threshold)[0] for prob in probs])
        
        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            centroids=gm.means_,
            probabilities=probs
        )

class HierarchicalClusterer(BaseClusterer):
    """层次聚类"""
    def __init__(self, n_clusters: Optional[int] = None, distance_threshold: Optional[float] = 0.5):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        
    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold if self.n_clusters is None else None
        )
        labels = clustering.fit_predict(embeddings)
        
        return ClusteringResult(
            labels=labels,
            n_clusters=len(np.unique(labels)),
            centroids=None  # 层次聚类没有明确的中心点
        )

class DBSCANClusterer(BaseClusterer):
    """基于密度的聚类"""
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(embeddings)
        
        # 计算每个簇的中心点
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            if label != -1:  # 排除噪声点
                mask = labels == label
                centroids.append(embeddings[mask].mean(axis=0))
                
        return ClusteringResult(
            labels=labels,
            n_clusters=len(np.unique(labels[labels != -1])),  # 不计入噪声点
            centroids=np.array(centroids) if centroids else None
        )

class RaptorClusterer(BaseClusterer):
    """RAPTOR论文中的聚类方法"""
    def __init__(self, dim: int = 10, threshold: float = 0.1):
        self.dim = dim # 降维后的维度
        self.threshold = threshold # 聚类阈值
        self._validate_params()
    
    def _validate_params(self):
        """验证参数有效性"""
        if self.dim <= 0:
            raise ValueError("Dimension must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
    
    def _global_cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """全局降维"""
        # print("embeddings.shape:", embeddings.shape) # embeddings.shape: (18, 3072)
        n_neighbors = int((len(embeddings) - 1) ** 0.5) # sqrt(嵌入向量的数量-1)
        result = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=self.dim,
            metric="cosine"
        ).fit_transform(embeddings)
        # print("result.shape:", result.shape) # result.shape: (18, 10)
        return result
        
    def _local_cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """局部降维"""
        return umap.UMAP(
            n_neighbors=10,
            n_components=self.dim,
            metric="cosine"
        ).fit_transform(embeddings)
    
    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        if len(embeddings) <= self.dim + 1: # 如果嵌入向量的数量小于等于降维后的维度加1，则直接返回一个全0的标签 - 这为啥？ - 是因为降维后的数据点不足以形成簇，例如：8个数据点，降到10维，降维后的数据点不足以形成簇，所以直接返回一个全0的标签 - 还是理解的不是清楚，但这里是没有问题的。
            return ClusteringResult(
                labels=np.array([0] * len(embeddings)),
                n_clusters=1
            )
            
        # 全局降维和聚类
        reduced_embeddings_global = self._global_cluster_embeddings(embeddings)
        gmm_clusterer = GMMClusterer(threshold=self.threshold)
        global_result = gmm_clusterer.fit_predict(reduced_embeddings_global)
        # print("global_result.labels:", global_result.labels)

        # 局部聚类
        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0
        
        for i in range(global_result.n_clusters): # 几个簇
            # 提取当前全局簇的嵌入
            mask = np.array([i in gc for gc in global_result.labels]) # 类似：[True, False, True, ...]，找到每个簇对应的是哪几个嵌入向量
            global_cluster_embeddings = embeddings[mask] # 找到这个簇对应的嵌入向量。global_cluster_embeddings.shape: (n, 3072)，n是这个簇的嵌入向量的数量
            
            if len(global_cluster_embeddings) <= self.dim + 1:
                local_labels = np.array([0] * len(global_cluster_embeddings))
                n_local_clusters = 1
            else: # 如果全局聚类之后的一个簇中的嵌入向量个数过多，则再次进行聚类
                # 局部降维和聚类
                reduced_local = self._local_cluster_embeddings(global_cluster_embeddings)
                local_result = gmm_clusterer.fit_predict(reduced_local)
                local_labels = local_result.labels
                n_local_clusters = local_result.n_clusters
                
            # 更新标签
            indices = np.where(mask)[0] # 返回mask中为True的索引。np.where(mask)是<class 'tuple'>，np.where(mask)[0]是<class 'numpy.ndarray'>
            for idx, local_label in zip(indices, local_labels):
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx],
                    local_label + total_clusters
                )
                
            total_clusters += n_local_clusters
            
        return ClusteringResult(
            labels=np.array(all_local_clusters),
            n_clusters=total_clusters
        )

class ClusteringFactory:
    """聚类算法工厂类"""
    @staticmethod # staticmethod：静态方法，不需要实例化类即可调用
    def get_clusterer(config: RaptorConfig) -> BaseClusterer:
        """
        根据配置创建聚类器
        Args:
            config: RaptorConfig配置对象
        Returns:
            BaseClusterer: 聚类器实例
        """
        clusterers = {
            'gmm': GMMClusterer,
            'hierarchical': HierarchicalClusterer,
            'dbscan': DBSCANClusterer,
            'raptor': RaptorClusterer
        }
        
        if config.clustering_method not in clusterers:
            raise ValueError(f"Unsupported clustering method: {config.clustering_method}")
            
        # 获取当前聚类方法的参数
        params = config.get_clustering_params() # 根据聚类方法到reptor_config.py中获取指定聚类方法的参数（参数在reptor_config.py中指定） # 类似：{'threshold': self.gmm_threshold}
        return clusterers[config.clustering_method](**params) # 类似：GMMClusterer(threshold=self.gmm_threshold)

# # 使用示例

# # 创建配置
# config = RaptorConfig(
#     enabled=True,
#     clustering_method='raptor',
#     n_levels=3,
#     dim_reduction=10,
#     raptor_threshold=0.1
# )

# # 创建聚类器
# clusterer = ClusteringFactory.get_clusterer(config)

# # 执行聚类
# result = clusterer.fit_predict(embeddings)