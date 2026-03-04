"""
用户行为分析模块 - Behavior Analyzer Module
负责页面停留、跳出率、访问深度、行为路径分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go


class BehaviorAnalyzer:
    """用户行为分析器"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
    
    def load_data(self, data: pd.DataFrame) -> None:
        """加载数据"""
        self.data = data
    
    def calculate_page_metrics(self, page_column: str = 'page', 
                               duration_column: str = 'duration') -> Dict[str, Any]:
        """计算页面指标"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        page_stats = self.data.groupby(page_column).agg(
            views=('user_id', 'count'),
            unique_visitors=('user_id', 'nunique'),
            avg_duration=(duration_column, 'mean') if duration_column in self.data.columns else ('user_id', 'count'),
            median_duration=(duration_column, 'median') if duration_column in self.data.columns else ('user_id', 'count')
        ).reset_index()
        
        return {
            'page_stats': page_stats,
            'most_viewed_page': page_stats.loc[page_stats['views'].idxmax(), page_column],
            'avg_time_on_page': page_stats['avg_duration'].mean()
        }
    
    def calculate_bounce_rate(self, session_column: str = 'session_id',
                             page_column: str = 'page') -> float:
        """计算跳出率（单页会话比例）"""
        if self.data is None or session_column not in self.data.columns:
            raise ValueError("数据无效或缺少会话列")
        
        session_counts = self.data.groupby(session_column)[page_column].count()
        bounce_sessions = (session_counts == 1).sum()
        total_sessions = len(session_counts)
        
        bounce_rate = (bounce_sessions / total_sessions * 100) if total_sessions > 0 else 0
        return bounce_rate
    
    def calculate_visit_depth(self, session_column: str = 'session_id') -> Dict[str, float]:
        """计算访问深度分布"""
        if self.data is None or session_column not in self.data.columns:
            raise ValueError("数据无效或缺少会话列")
        
        session_depths = self.data.groupby(session_column).size()
        
        depth_distribution = {
            'avg_depth': session_depths.mean(),
            'median_depth': session_depths.median(),
            'max_depth': session_depths.max(),
            'depth_1_percentage': (session_depths == 1).sum() / len(session_depths) * 100,
            'depth_2_5_percentage': ((session_depths >= 2) & (session_depths <= 5)).sum() / len(session_depths) * 100,
            'depth_5_plus_percentage': (session_depths > 5).sum() / len(session_depths) * 100
        }
        
        return depth_distribution
    
    def analyze_user_paths(self, session_column: str = 'session_id',
                          page_column: str = 'page',
                          timestamp_column: str = 'timestamp',
                          max_paths: int = 10) -> Dict[str, Any]:
        """分析用户行为路径"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        # 按会话和时间排序
        sorted_data = self.data.sort_values([session_column, timestamp_column])
        
        # 构建路径
        paths = []
        for session_id in sorted_data[session_column].unique():
            session_data = sorted_data[sorted_data[session_column] == session_id]
            path = session_data[page_column].tolist()
            if len(path) > 1:
                paths.append(' -> '.join(path[:5]))  # 限制路径长度
        
        # 统计最常见路径
        path_series = pd.Series(paths)
        top_paths = path_series.value_counts().head(max_paths)
        
        return {
            'top_paths': top_paths,
            'avg_path_length': np.mean([len(p.split(' -> ')) for p in paths]) if paths else 0,
            'total_unique_paths': len(set(paths))
        }
    
    def create_sankey_data(self, session_column: str = 'session_id',
                          page_column: str = 'page',
                          timestamp_column: str = 'timestamp') -> Dict[str, Any]:
        """准备桑基图数据"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        sorted_data = self.data.sort_values([session_column, timestamp_column])
        
        sources = []
        targets = []
        values = []
        
        for session_id in sorted_data[session_column].unique():
            session_data = sorted_data[sorted_data[session_column] == session_id]
            pages = session_data[page_column].tolist()
            
            for i in range(len(pages) - 1):
                sources.append(pages[i])
                targets.append(pages[i + 1])
                values.append(1)
        
        # 聚合
        flow_df = pd.DataFrame({'source': sources, 'target': targets, 'value': values})
        flow_df = flow_df.groupby(['source', 'target']).sum().reset_index()
        
        return {
            'sources': flow_df['source'].tolist(),
            'targets': flow_df['target'].tolist(),
            'values': flow_df['value'].tolist()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取行为分析摘要"""
        return {
            'bounce_rate': self.calculate_bounce_rate() if self.data is not None else None,
            'visit_depth': self.calculate_visit_depth() if self.data is not None else None,
            'data_shape': self.data.shape if self.data is not None else None
        }
