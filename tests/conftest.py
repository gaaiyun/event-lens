"""
测试配置文件 - Test Configuration
共享测试夹具和工具
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_traffic_data():
    """生成示例流量数据"""
    np.random.seed(42)
    n_records = 1000
    
    dates = [datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 30)) for _ in range(n_records)]
    
    data = {
        'date': dates,
        'user_id': np.random.randint(1, 201, n_records),
        'session_id': [f'sess_{i}' for i in np.random.randint(1, 501, n_records)],
        'page': np.random.choice(['home', 'product', 'cart', 'checkout'], n_records),
        'source': np.random.choice(['google', 'facebook', 'direct', 'email'], n_records),
        'device': np.random.choice(['desktop', 'mobile', 'tablet'], n_records),
        'country': np.random.choice(['China', 'USA', 'UK'], n_records),
        'duration': np.random.exponential(120, n_records).clip(5, 600),
        'timestamp': [dates[i] + timedelta(minutes=i) for i in range(n_records)]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_cohort_data():
    """生成示例 Cohort 数据"""
    np.random.seed(42)
    n_users = 500
    
    data = {
        'user_id': range(1, n_users + 1),
        'date': [datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 60)) for _ in range(n_users)],
        'recency': np.random.randint(1, 60, n_users),
        'frequency': np.random.randint(1, 20, n_users),
        'monetary': np.random.exponential(500, n_users)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_series_data():
    """生成示例时间序列数据"""
    np.random.seed(42)
    n_days = 100
    
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    data = {
        'date': dates,
        'visits': np.random.poisson(1000, n_days) + np.sin(np.arange(n_days) / 7) * 200
    }
    
    return pd.DataFrame(data)
