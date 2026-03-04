"""
流量分析模块测试 - Traffic Analyzer Tests
"""

import pytest
import pandas as pd
import numpy as np
from traffic_analyzer import TrafficAnalyzer


class TestTrafficAnalyzer:
    """流量分析器测试类"""
    
    def test_initialization(self):
        """测试初始化"""
        analyzer = TrafficAnalyzer()
        assert analyzer.data is None
        assert analyzer.metrics == {}
    
    def test_load_data(self, sample_traffic_data):
        """测试数据加载"""
        analyzer = TrafficAnalyzer()
        analyzer.load_data(sample_traffic_data)
        assert analyzer.data is not None
        assert len(analyzer.data) == len(sample_traffic_data)
    
    def test_calculate_pv_uv(self, sample_traffic_data):
        """测试 PV/UV 计算"""
        analyzer = TrafficAnalyzer(sample_traffic_data)
        result = analyzer.calculate_pv_uv()
        
        assert 'pv' in result
        assert 'uv' in result
        assert 'pv_uv_ratio' in result
        assert result['pv'] == len(sample_traffic_data)
        assert result['uv'] == sample_traffic_data['user_id'].nunique()
        assert result['pv_uv_ratio'] > 0
    
    def test_calculate_pv_uv_no_data(self):
        """测试无数据时的 PV/UV 计算"""
        analyzer = TrafficAnalyzer()
        with pytest.raises(ValueError):
            analyzer.calculate_pv_uv()
    
    def test_analyze_traffic_sources(self, sample_traffic_data):
        """测试流量来源分析"""
        analyzer = TrafficAnalyzer(sample_traffic_data)
        result = analyzer.analyze_traffic_sources('source')
        
        assert 'source_distribution' in result
        assert 'top_source' in result
        assert 'source_count' in result
        assert result['source_count'] > 0
        assert result['top_source'] is not None
    
    def test_analyze_traffic_sources_no_column(self, sample_traffic_data):
        """测试缺少来源列时的处理"""
        analyzer = TrafficAnalyzer(sample_traffic_data)
        with pytest.raises(ValueError):
            analyzer.analyze_traffic_sources('nonexistent_column')
    
    def test_analyze_devices(self, sample_traffic_data):
        """测试设备分析"""
        analyzer = TrafficAnalyzer(sample_traffic_data)
        result = analyzer.analyze_devices('device')
        
        assert 'device_distribution' in result
        assert 'mobile_percentage' in result
        assert result['mobile_percentage'] >= 0
        assert result['mobile_percentage'] <= 100
    
    def test_analyze_geography(self, sample_traffic_data):
        """测试地域分析"""
        analyzer = TrafficAnalyzer(sample_traffic_data)
        result = analyzer.analyze_geography('country')
        
        assert 'geo_distribution' in result
        assert 'top_countries' in result
        assert 'country_count' in result
        assert result['country_count'] > 0
    
    def test_get_summary(self, sample_traffic_data):
        """测试摘要获取"""
        analyzer = TrafficAnalyzer(sample_traffic_data)
        analyzer.calculate_pv_uv()
        result = analyzer.get_summary()
        
        assert 'metrics' in result
        assert 'data_shape' in result
        assert 'columns' in result
        assert result['data_shape'] == sample_traffic_data.shape
    
    def test_quality_score_calculation(self, sample_traffic_data):
        """测试来源质量评分计算"""
        analyzer = TrafficAnalyzer(sample_traffic_data)
        result = analyzer.analyze_traffic_sources('source')
        
        source_dist = result['source_distribution']
        assert 'quality_score' in source_dist.columns
        assert source_dist['quality_score'].min() >= 0
        assert source_dist['quality_score'].max() <= 1
    
    def test_daily_stats_calculation(self, sample_traffic_data):
        """测试每日统计计算"""
        analyzer = TrafficAnalyzer(sample_traffic_data)
        result = analyzer.calculate_pv_uv('date')
        
        assert result['daily_stats'] is not None
        assert len(result['daily_stats']) > 0
        assert 'pv' in result['daily_stats'].columns
        assert 'uv' in result['daily_stats'].columns
