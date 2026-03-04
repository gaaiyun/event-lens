"""
留存分析模块 - Retention Analyzer Module
负责 Cohort 分析、留存曲线、留存驱动因素
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class RetentionAnalyzer:
    """留存分析器"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
        self.cohort_data = None
    
    def load_data(self, data: pd.DataFrame) -> None:
        """加载数据"""
        self.data = data
    
    def create_cohorts(self, date_column: str = 'date',
                      user_column: str = 'user_id',
                      cohort_period: str = 'week') -> pd.DataFrame:
        """创建 Cohort 分组"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        data = self.data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        
        # 获取用户首次访问日期
        first_visit = data.groupby(user_column)[date_column].min().reset_index()
        first_visit.columns = [user_column, 'cohort_date']
        
        # 按周期分组
        if cohort_period == 'week':
            first_visit['cohort'] = first_visit['cohort_date'].dt.to_period('W').dt.start_time
        elif cohort_period == 'month':
            first_visit['cohort'] = first_visit['cohort_date'].dt.to_period('M').dt.start_time
        else:
            first_visit['cohort'] = first_visit['cohort_date'].dt.date
        
        # 合并回原数据
        data = data.merge(first_visit, on=user_column)
        
        # 计算周期索引
        data['period'] = (data[date_column] - data['cohort']).dt.days
        
        if cohort_period == 'week':
            data['period'] = data['period'] // 7
        elif cohort_period == 'month':
            data['period'] = data['period'] // 30
        
        self.cohort_data = data
        return data
    
    def calculate_retention_matrix(self, user_column: str = 'user_id',
                                  period_column: str = 'period') -> pd.DataFrame:
        """计算留存矩阵"""
        if self.cohort_data is None:
            raise ValueError("未创建 Cohort 数据")
        
        # 计算每个 cohort 在每个周期的用户数
        cohort_counts = self.cohort_data.groupby(['cohort', period_column])[user_column].nunique().unstack(fill_value=0)
        
        # 计算留存率
        cohort_sizes = cohort_counts.iloc[:, 0]
        retention_matrix = cohort_counts.divide(cohort_sizes, axis=0) * 100
        
        return retention_matrix.round(2)
    
    def calculate_retention_rates(self, retention_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算关键留存率"""
        if retention_matrix is None or retention_matrix.empty:
            return {}
        
        # 次日留存 (period 1)
        day1_retention = retention_matrix.iloc[:, 1].mean() if len(retention_matrix.columns) > 1 else 0
        
        # 7 日留存 (period 7 或最接近的)
        day7_col = min(7, len(retention_matrix.columns) - 1)
        day7_retention = retention_matrix.iloc[:, day7_col].mean() if len(retention_matrix.columns) > day7_col else 0
        
        # 30 日留存
        day30_col = min(30, len(retention_matrix.columns) - 1)
        day30_retention = retention_matrix.iloc[:, day30_col].mean() if len(retention_matrix.columns) > day30_col else 0
        
        return {
            'day1_retention': round(day1_retention, 2),
            'day7_retention': round(day7_retention, 2),
            'day30_retention': round(day30_retention, 2)
        }
    
    def analyze_retention_drivers(self, feature_columns: List[str],
                                 user_column: str = 'user_id',
                                 retained_days: int = 7) -> pd.DataFrame:
        """分析留存驱动因素"""
        if self.cohort_data is None:
            raise ValueError("未创建 Cohort 数据")
        
        # 标记留存用户
        retained_users = self.cohort_data[self.cohort_data['period'] >= retained_days][user_column].unique()
        self.cohort_data['retained'] = self.cohort_data[user_column].isin(retained_users)
        
        # 分析各特征与留存的关系
        driver_analysis = []
        for col in feature_columns:
            if col in self.cohort_data.columns:
                grouped = self.cohort_data.groupby(col)['retained'].mean() * 100
                driver_analysis.append({
                    'feature': col,
                    'retention_by_group': grouped.to_dict()
                })
        
        return pd.DataFrame(driver_analysis)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取留存分析摘要"""
        if self.cohort_data is None:
            return {'status': '未初始化'}
        
        retention_matrix = self.calculate_retention_matrix()
        retention_rates = self.calculate_retention_rates(retention_matrix)
        
        return {
            'retention_rates': retention_rates,
            'cohort_count': len(retention_matrix),
            'matrix_shape': retention_matrix.shape
        }
