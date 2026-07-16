"""
留存分析模块测试 - Retention Analyzer Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from retention_analyzer import RetentionAnalyzer


class TestRetentionAnalyzer:
    """留存分析器测试类"""
    
    def test_initialization(self):
        """测试初始化"""
        analyzer = RetentionAnalyzer()
        assert analyzer.data is None
        assert analyzer.cohort_data is None
    
    def test_load_data(self, sample_traffic_data):
        """测试数据加载"""
        analyzer = RetentionAnalyzer()
        analyzer.load_data(sample_traffic_data)
        assert analyzer.data is not None
    
    def test_create_cohorts_weekly(self, sample_traffic_data):
        """测试按周创建 Cohort"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        cohort_data = analyzer.create_cohorts('date', 'user_id', 'week')
        
        assert analyzer.cohort_data is not None
        assert 'cohort' in cohort_data.columns
        assert 'period' in cohort_data.columns
        assert len(cohort_data) == len(sample_traffic_data)
    
    def test_create_cohorts_monthly(self, sample_traffic_data):
        """测试按月创建 Cohort"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        cohort_data = analyzer.create_cohorts('date', 'user_id', 'month')
        
        assert 'cohort' in cohort_data.columns
        assert 'period' in cohort_data.columns
    
    def test_create_cohorts_no_data(self):
        """测试无数据时创建 Cohort"""
        analyzer = RetentionAnalyzer()
        with pytest.raises(ValueError):
            analyzer.create_cohorts()
    
    def test_calculate_retention_matrix(self, sample_traffic_data):
        """测试留存矩阵计算"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        analyzer.create_cohorts('date', 'user_id', 'week')
        retention_matrix = analyzer.calculate_retention_matrix()
        
        assert isinstance(retention_matrix, pd.DataFrame)
        assert len(retention_matrix) > 0
        # 留存率应在 0-100 之间
        assert retention_matrix.min().min() >= 0
        assert retention_matrix.max().max() <= 100
    
    def test_calculate_retention_matrix_no_cohort(self, sample_traffic_data):
        """测试未创建 Cohort 时的留存矩阵"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        with pytest.raises(ValueError):
            analyzer.calculate_retention_matrix()
    
    def test_calculate_retention_rates(self, sample_traffic_data):
        """测试留存率计算"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        analyzer.create_cohorts('date', 'user_id', 'week')
        retention_matrix = analyzer.calculate_retention_matrix()
        rates = analyzer.calculate_retention_rates(retention_matrix)
        
        assert rates['granularity'] == 'week'
        assert 'week1_retention' in rates
        assert 'week4_retention' in rates
        assert 'week12_retention' in rates
        assert all(
            0 <= rates[key] <= 100
            for key in ('week1_retention', 'week4_retention', 'week12_retention')
        )

    def test_retention_rates_use_exact_period_labels(self):
        """缺失 day 1 时不能把 day 2 的列位置冒充次日留存。"""
        analyzer = RetentionAnalyzer()
        analyzer.cohort_period = 'day'
        retention_matrix = pd.DataFrame(
            {0: [100.0, 100.0], 2: [50.0, 25.0], 7: [20.0, 10.0]}
        )

        rates = analyzer.calculate_retention_rates(retention_matrix)

        assert rates['granularity'] == 'day'
        assert rates['day1_retention'] == 0.0
        assert rates['day7_retention'] == 15.0
        assert rates['day30_retention'] == 0.0
    
    def test_analyze_retention_drivers(self, sample_traffic_data):
        """测试留存驱动因素分析"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        analyzer.create_cohorts('date', 'user_id', 'week')
        
        feature_cols = ['source', 'device', 'country']
        result = analyzer.analyze_retention_drivers(feature_cols, 'user_id', 7)
        
        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'retention_by_group' in result.columns
    
    def test_get_summary(self, sample_traffic_data):
        """测试摘要获取"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        analyzer.create_cohorts('date', 'user_id', 'week')
        result = analyzer.get_summary()
        
        assert 'retention_rates' in result
        assert 'cohort_count' in result
        assert 'matrix_shape' in result
    
    def test_get_summary_not_initialized(self):
        """测试未初始化时的摘要"""
        analyzer = RetentionAnalyzer()
        result = analyzer.get_summary()
        
        assert 'status' in result
        assert result['status'] == '未初始化'
    
    def test_cohort_period_calculation(self, sample_traffic_data):
        """测试 Cohort 周期计算"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        cohort_data = analyzer.create_cohorts('date', 'user_id', 'week')
        
        # period 应为非负数
        assert (cohort_data['period'] >= 0).all()
    
    def test_retention_matrix_first_period(self, sample_traffic_data):
        """测试留存矩阵首期留存率"""
        analyzer = RetentionAnalyzer(sample_traffic_data)
        analyzer.create_cohorts('date', 'user_id', 'week')
        retention_matrix = analyzer.calculate_retention_matrix()
        
        # 首期留存率应接近 100%
        first_period = retention_matrix.iloc[:, 0]
        assert first_period.min() >= 95  # 允许小误差
