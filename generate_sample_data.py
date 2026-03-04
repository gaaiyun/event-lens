"""
生成示例数据脚本
Generate comprehensive sample data for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_comprehensive_sample_data(n_records=10000, output_file='sample_data.csv'):
    """生成综合示例数据"""
    np.random.seed(42)
    
    # 日期范围
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range = (end_date - start_date).days
    
    # 生成基础数据
    dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_records)]
    
    # 用户 ID（2000 个独立用户）
    user_ids = np.random.randint(1, 2001, n_records)
    
    # 会话 ID（5000 个会话）
    session_ids = [f'sess_{i}' for i in np.random.randint(1, 5001, n_records)]
    
    # 页面（带权重）
    pages = np.random.choice(
        ['home', 'product', 'cart', 'checkout', 'about', 'contact', 'blog'],
        n_records,
        p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05]
    )
    
    # 流量来源（带权重）
    sources = np.random.choice(
        ['google', 'facebook', 'direct', 'email', 'twitter', 'linkedin'],
        n_records,
        p=[0.35, 0.25, 0.20, 0.10, 0.05, 0.05]
    )
    
    # 设备（带权重）
    devices = np.random.choice(
        ['desktop', 'mobile', 'tablet'],
        n_records,
        p=[0.50, 0.40, 0.10]
    )
    
    # 国家/地区（带权重）
    countries = np.random.choice(
        ['China', 'USA', 'UK', 'Japan', 'Germany', 'France', 'Other'],
        n_records,
        p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05]
    )
    
    # 停留时长（指数分布，模拟真实用户行为）
    durations = np.random.exponential(120, n_records).clip(5, 600).round(1)
    
    # 时间戳
    timestamps = [dates[i] + timedelta(minutes=i*5 % 1440) for i in range(n_records)]
    
    # 创建 DataFrame
    df = pd.DataFrame({
        'date': dates,
        'user_id': user_ids,
        'session_id': session_ids,
        'page': pages,
        'source': sources,
        'device': devices,
        'country': countries,
        'duration': durations,
        'timestamp': timestamps
    })
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 保存
    df.to_csv(output_file, index=False)
    print(f"✓ 已生成 {n_records:,} 条示例数据")
    print(f"✓ 文件保存至：{output_file}")
    print(f"\n数据统计:")
    print(f"  - 独立用户数：{df['user_id'].nunique():,}")
    print(f"  - 总会话数：{df['session_id'].nunique():,}")
    print(f"  - 日期范围：{df['date'].min()} 至 {df['date'].max()}")
    print(f"  - 平均停留时长：{df['duration'].mean():.1f}秒")
    
    return df


if __name__ == "__main__":
    generate_comprehensive_sample_data()
