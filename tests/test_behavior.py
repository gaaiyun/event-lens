"""
用户行为分析模块测试 - Behavior Analyzer Tests
"""

import pytest
import pandas as pd
import numpy as np
from behavior_analyzer import BehaviorAnalyzer


class TestBehaviorAnalyzer:
    """用户行为分析器测试类"""
    
    def test_initialization(self):
        """测试初始化"""
        analyzer = BehaviorAnalyzer()
        assert analyzer.data is None
    
    def test_load_data(self, sample_traffic_data):
        """测试数据加载"""
        analyzer = BehaviorAnalyzer()
        analyzer.load_data(sample_traffic_data)
        assert analyzer.data is not None
    
    def test_calculate_page_metrics(self, sample_traffic_data):
        """测试页面指标计算"""
        analyzer = BehaviorAnalyzer(sample_traffic_data)
        result = analyzer.calculate_page_metrics('page', 'duration')
        
        assert 'page_stats' in result
        assert 'most_viewed_page' in result
        assert 'avg_time_on_page' in result
        assert len(result['page_stats']) > 0
    
    def test_calculate_bounce_rate(self, sample_traffic_data):
        """测试跳出率计算"""
        analyzer = BehaviorAnalyzer(sample_traffic_data)
        bounce_rate = analyzer.calculate_bounce_rate('session_id', 'page')
        
        assert isinstance(bounce_rate, float)
        assert bounce_rate >= 0
        assert bounce_rate <= 100
    
    def test_calculate_bounce_rate_no_data(self):
        """测试无数据时的跳出率计算"""
        analyzer = BehaviorAnalyzer()
        with pytest.raises(ValueError):
            analyzer.calculate_bounce_rate()
    
    def test_calculate_visit_depth(self, sample_traffic_data):
        """测试访问深度计算"""
        analyzer = BehaviorAnalyzer(sample_traffic_data)
        result = analyzer.calculate_visit_depth('session_id')
        
        assert 'avg_depth' in result
        assert 'median_depth' in result
        assert 'max_depth' in result
        assert 'depth_1_percentage' in result
        assert result['avg_depth'] > 0
        assert result['max_depth'] >= result['avg_depth']
    
    def test_analyze_user_paths(self, sample_traffic_data):
        """测试用户路径分析"""
        analyzer = BehaviorAnalyzer(sample_traffic_data)
        result = analyzer.analyze_user_paths('session_id', 'page', 'timestamp')
        
        assert 'top_paths' in result
        assert 'avg_path_length' in result
        assert 'total_unique_paths' in result
        assert result['avg_path_length'] >= 1
    
    def test_create_sankey_data(self, sample_traffic_data):
        """测试桑基图数据生成"""
        analyzer = BehaviorAnalyzer(sample_traffic_data)
        result = analyzer.create_sankey_data('session_id', 'page', 'timestamp')
        
        assert 'sources' in result
        assert 'targets' in result
        assert 'values' in result
        assert len(result['sources']) == len(result['targets'])
        assert len(result['sources']) == len(result['values'])
    
    def test_get_summary(self, sample_traffic_data):
        """测试摘要获取"""
        analyzer = BehaviorAnalyzer(sample_traffic_data)
        result = analyzer.get_summary()
        
        assert 'bounce_rate' in result
        assert 'visit_depth' in result
        assert 'data_shape' in result
        assert result['bounce_rate'] is not None
        assert result['visit_depth'] is not None
    
    def test_page_metrics_with_missing_duration(self, sample_traffic_data):
        """测试缺少停留时长列时的页面指标"""
        data = sample_traffic_data.drop(columns=['duration'])
        analyzer = BehaviorAnalyzer(data)
        result = analyzer.calculate_page_metrics('page')
        
        assert 'page_stats' in result
        assert len(result['page_stats']) > 0
    
    def test_visit_depth_distribution(self, sample_traffic_data):
        """测试访问深度分布合理性"""
        analyzer = BehaviorAnalyzer(sample_traffic_data)
        result = analyzer.calculate_visit_depth('session_id')
        
        # 各深度百分比之和应接近 100
        total_percentage = (
            result['depth_1_percentage'] +
            result['depth_2_5_percentage'] +
            result['depth_5_plus_percentage']
        )
        assert 95 <= total_percentage <= 105  # 允许小误差
