from dataclasses import dataclass
from typing import Optional, Literal # Optional：表示参数是可选的，可以为None。Literal：表示参数是固定的，只能是给定的值之一。

@dataclass
class RaptorConfig:
    """RAPTOR 配置类"""
    enabled: bool = False
    clustering_method: Literal['gmm', 'hierarchical', 'dbscan', 'raptor'] = 'raptor'  # 聚类方法
    n_clusters: Optional[int] = 5  # 每层最多聚几个类别（某些算法可能不需要）
    n_levels: int = 2    # 层级数量（包含原始文档层） - n_levels等于多少就聚几次
    min_cluster_size: int = 3  # 最小聚类大小：如果一个聚类中的文档数量小于这个值，则该聚类将被合并到其他聚类中。
    
    # 聚类算法特定参数
    # GMM参数
    gmm_threshold: float = 0.1  # GMM概率阈值。具体来说，这个参数是用于确定一个点属于某个聚类的概率阈值。如果一个点属于某个聚类的概率大于这个阈值，那么这个点就被认为是属于这个聚类。 - 存疑
    
    # DBSCAN参数
    dbscan_eps: float = 0.5  # DBSCAN邻域范围
    dbscan_min_samples: int = 5  # DBSCAN最小样本数
    
    # 层次聚类参数
    hierarchical_distance_threshold: Optional[float] = 0.5  # 层次聚类距离阈值
    
    # RAPTOR特定参数
    dim_reduction: int = 10  # UMAP降维维度
    raptor_threshold: float = 0.1  # RAPTOR聚类阈值
    
    def get_clustering_params(self) -> dict:
        """获取当前聚类方法的相关参数"""
        params = {
            'gmm': {
                'threshold': self.gmm_threshold
            },
            'dbscan': {
                'eps': self.dbscan_eps,
                'min_samples': self.dbscan_min_samples
            },
            'hierarchical': {
                'n_clusters': self.n_clusters,
                'distance_threshold': self.hierarchical_distance_threshold
            },
            'raptor': {
                'dim': self.dim_reduction,
                'threshold': self.raptor_threshold
            }
        }
        return params[self.clustering_method] 