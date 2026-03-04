"""
细分分析模块 - Segmentation Analyzer Module
负责多维度用户细分和对比分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class SegmentationAnalyzer:
    """用户细分分析器"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
        self.segments = None
    
    def load_data(self, data: pd.DataFrame) -> None:
        """加载数据"""
        self.data = data
    
    def segment_by_dimension(self, dimension_columns: List[str],
                            metric_columns: List[str]) -> pd.DataFrame:
        """按维度细分并计算指标"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        available_dims = [col for col in dimension_columns if col in self.data.columns]
        available_metrics = [col for col in metric_columns if col in self.data.columns]
        
        if len(available_dims) == 0 or len(available_metrics) == 0:
            raise ValueError("没有可用的维度或指标列")
        
        # 分组聚合
        grouped = self.data.groupby(available_dims)[available_metrics].agg([
            'sum', 'mean', 'count', 'std'
        ])
        
        return grouped
    
    def create_user_segments(self, feature_columns: List[str],
                            n_clusters: int = 5,
                            user_column: str = 'user_id') -> Dict[str, Any]:
        """使用 K-Means 创建用户分群"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        available_cols = [col for col in feature_columns if col in self.data.columns]
        if len(available_cols) == 0:
            raise ValueError("没有可用的特征列")
        
        # 按用户聚合
        user_features = self.data.groupby(user_column)[available_cols].agg(['sum', 'mean', 'count'])
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns]
        
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(user_features)
        
        # K-Means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        user_features['segment'] = clusters
        
        # 分析各分群特征
        segment_profiles = user_features.groupby('segment').mean()
        
        self.segments = user_features
        
        return {
            'segment_profiles': segment_profiles,
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'inertia': kmeans.inertia_
        }
    
    def compare_segments(self, segment_column: str = 'segment',
                        metric_columns: List[str] = None) -> pd.DataFrame:
        """对比不同细分群体"""
        if self.segments is None:
            raise ValueError("未创建细分数据")
        
        if metric_columns is None:
            metric_columns = [col for col in self.segments.columns if col != segment_column]
        
        comparison = self.segments.groupby(segment_column)[metric_columns].agg([
            'mean', 'median', 'std', 'count'
        ])
        
        return comparison
    
    def rfm_segmentation(self, recency_col: str = 'recency',
                        frequency_col: str = 'frequency',
                        monetary_col: str = 'monetary',
                        n_bins: int = 5) -> pd.DataFrame:
        """RFM 用户价值细分"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        data = self.data.copy()
        
        # 分位数分箱
        data['R_score'] = pd.qcut(data[recency_col], n_bins, labels=False, duplicates='drop')
        data['F_score'] = pd.qcut(data[frequency_col], n_bins, labels=False, duplicates='drop')
        data['M_score'] = pd.qcut(data[monetary_col], n_bins, labels=False, duplicates='drop')
        
        # 计算 RFM 总分
        data['RFM_score'] = data['R_score'] + data['F_score'] + data['M_score']
        
        # 划分用户群体
        def rfm_segment(score):
            if score >= 12:
                return 'Champions'
            elif score >= 9:
                return 'Loyal Customers'
            elif score >= 6:
                return 'Potential Loyalists'
            elif score >= 3:
                return 'At Risk'
            else:
                return 'Lost'
        
        data['RFM_segment'] = data['RFM_score'].apply(rfm_segment)
        
        return data
    
    def get_summary(self) -> Dict[str, Any]:
        """获取细分分析摘要"""
        return {
            'segments_created': self.segments is not None,
            'segment_count': self.segments['segment'].nunique() if self.segments is not None else 0,
            'data_shape': self.data.shape if self.data is not None else None
        }
