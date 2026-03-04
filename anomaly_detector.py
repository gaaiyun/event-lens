"""
异常检测模块 - Anomaly Detector Module
负责流量异常自动检测、统计告警
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
        self.anomalies = []
        self.alerts = []
    
    def load_data(self, data: pd.DataFrame) -> None:
        """加载数据"""
        self.data = data
    
    def detect_statistical_anomalies(self, metric_column: str = 'visits',
                                     threshold: float = 3.0) -> Dict[str, Any]:
        """使用统计方法检测异常（Z-Score）"""
        if self.data is None or metric_column not in self.data.columns:
            raise ValueError("数据无效")
        
        values = self.data[metric_column].values
        mean = np.mean(values)
        std = np.std(values)
        
        # 计算 Z-Score
        z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros(len(values))
        
        # 标记异常
        anomalies = z_scores > threshold
        anomaly_indices = np.where(anomalies)[0]
        
        self.anomalies = anomaly_indices.tolist()
        
        return {
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_values': values[anomalies].tolist() if len(anomalies) > 0 else [],
            'threshold': threshold,
            'mean': mean,
            'std': std
        }
    
    def detect_isolation_forest_anomalies(self, feature_columns: List[str],
                                         contamination: float = 0.1) -> Dict[str, Any]:
        """使用孤立森林检测异常"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        available_cols = [col for col in feature_columns if col in self.data.columns]
        if len(available_cols) == 0:
            raise ValueError("没有可用的特征列")
        
        X = self.data[available_cols].values
        
        # 训练孤立森林
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X)
        
        # -1 表示异常，1 表示正常
        anomalies = np.where(predictions == -1)[0]
        
        self.anomalies = anomalies.tolist()
        
        return {
            'anomaly_count': len(anomalies),
            'anomaly_indices': anomalies.tolist(),
            'contamination': contamination,
            'features_used': available_cols
        }
    
    def detect_trend_anomalies(self, date_column: str = 'date',
                              metric_column: str = 'visits',
                              window: int = 7) -> Dict[str, Any]:
        """检测趋势异常（移动平均偏离）"""
        if self.data is None:
            raise ValueError("未加载数据")
        
        data = self.data.sort_values(date_column).copy()
        data['moving_avg'] = data[metric_column].rolling(window=window, min_periods=1).mean()
        data['deviation'] = data[metric_column] - data['moving_avg']
        data['deviation_pct'] = (data['deviation'] / data['moving_avg'] * 100).replace([np.inf, -np.inf], 0)
        
        # 标记显著偏离（>50%）
        significant_deviations = data[np.abs(data['deviation_pct']) > 50]
        
        return {
            'anomaly_count': len(significant_deviations),
            'anomalies': significant_deviations[[date_column, metric_column, 'deviation_pct']].to_dict('records'),
            'avg_deviation': data['deviation_pct'].mean()
        }
    
    def create_alerts(self, alert_threshold: float = 2.5) -> List[Dict[str, Any]]:
        """生成告警"""
        self.alerts = []
        
        for idx in self.anomalies:
            if idx < len(self.data):
                row = self.data.iloc[idx]
                severity = 'high' if idx in self.anomalies[:3] else 'medium'
                
                self.alerts.append({
                    'index': idx,
                    'severity': severity,
                    'message': f"检测到异常：索引 {idx}",
                    'data': row.to_dict() if hasattr(row, 'to_dict') else str(row)
                })
        
        return self.alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """获取异常检测摘要"""
        return {
            'total_anomalies': len(self.anomalies),
            'total_alerts': len(self.alerts),
            'anomaly_rate': len(self.anomalies) / len(self.data) * 100 if self.data is not None else 0
        }
