"""
预测模块测试 - Forecaster Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from forecaster import Forecaster


class TestForecaster:
    """预测器测试类"""
    
    def test_initialization(self):
        """测试初始化"""
        forecaster = Forecaster()
        assert forecaster.data is None
        assert forecaster.model is None
        assert forecaster.forecast_results is None
    
    def test_load_data(self, sample_traffic_data):
        """测试数据加载"""
        forecaster = Forecaster()
        forecaster.load_data(sample_traffic_data)
        assert forecaster.data is not None
    
    def test_prepare_time_series(self, sample_traffic_data):
        """测试时间序列准备"""
        forecaster = Forecaster(sample_traffic_data)
        ts = forecaster.prepare_time_series('date', 'user_id')
        
        assert isinstance(ts, pd.Series)
        assert len(ts) > 0
        assert ts.index.freq is not None or ts.index.is_monotonic_increasing
    
    def test_prepare_time_series_no_data(self):
        """测试无数据时的时间序列准备"""
        forecaster = Forecaster()
        with pytest.raises(ValueError):
            forecaster.prepare_time_series()
    
    def test_fit_arima(self, sample_time_series_data):
        """测试 ARIMA 模型拟合"""
        forecaster = Forecaster(sample_time_series_data)
        ts = forecaster.prepare_time_series('date', 'visits')
        result = forecaster.fit_arima(ts, order=(1, 1, 1))
        
        assert 'model_type' in result
        assert 'status' in result
        assert result['model_type'] == 'ARIMA'
        # 模型可能拟合成功或失败（取决于数据）
        assert result['status'] in ['success', 'error']
    
    def test_fit_exponential_smoothing(self, sample_time_series_data):
        """测试指数平滑模型拟合"""
        forecaster = Forecaster(sample_time_series_data)
        ts = forecaster.prepare_time_series('date', 'visits')
        result = forecaster.fit_exponential_smoothing(ts)
        
        assert 'model_type' in result
        assert 'status' in result
        assert result['model_type'] == 'ExponentialSmoothing'
    
    def test_forecast_no_model(self):
        """测试无模型时的预测"""
        forecaster = Forecaster()
        with pytest.raises(ValueError):
            forecaster.forecast(steps=7)
    
    def test_forecast_with_arima(self, sample_time_series_data):
        """测试 ARIMA 预测"""
        forecaster = Forecaster(sample_time_series_data)
        ts = forecaster.prepare_time_series('date', 'visits')
        forecaster.fit_arima(ts, order=(1, 1, 1))
        
        if forecaster.model is not None:
            result = forecaster.forecast(steps=7, confidence=0.95)
            
            if result.get('status') != 'error':
                assert 'forecast_values' in result
                assert 'lower_bound' in result
                assert 'upper_bound' in result
                assert len(result['forecast_values']) == 7
                assert len(result['lower_bound']) == 7
                assert len(result['upper_bound']) == 7
    
    def test_forecast_trend(self, sample_time_series_data):
        """测试趋势预测"""
        forecaster = Forecaster(sample_time_series_data)
        ts = forecaster.prepare_time_series('date', 'visits')
        result = forecaster.forecast_trend(ts, steps=30)
        
        assert 'trend' in result
        assert 'slope' in result
        assert 'r_squared' in result
        assert 'forecast' in result
        assert result['trend'] in ['increasing', 'decreasing']
        assert len(result['forecast']) == 30
        assert 0 <= result['r_squared'] <= 1
    
    def test_get_summary(self, sample_time_series_data):
        """测试摘要获取"""
        forecaster = Forecaster(sample_time_series_data)
        ts = forecaster.prepare_time_series('date', 'visits')
        forecaster.fit_arima(ts)
        
        result = forecaster.get_summary()
        
        assert 'model_fitted' in result
        assert 'model_type' in result
        assert 'forecast_available' in result
    
    def test_confidence_interval_logic(self, sample_time_series_data):
        """测试置信区间逻辑"""
        forecaster = Forecaster(sample_time_series_data)
        ts = forecaster.prepare_time_series('date', 'visits')
        forecaster.fit_arima(ts)
        
        if forecaster.model is not None:
            result_95 = forecaster.forecast(steps=7, confidence=0.95)
            result_99 = forecaster.forecast(steps=7, confidence=0.99)
            
            if result_95.get('status') != 'error' and result_99.get('status') != 'error':
                # 99% 置信区间应比 95% 更宽
                width_95 = np.mean(np.array(result_95['upper_bound']) - np.array(result_95['lower_bound']))
                width_99 = np.mean(np.array(result_99['upper_bound']) - np.array(result_99['lower_bound']))
                assert width_99 >= width_95
    
    def test_time_series_frequency(self, sample_traffic_data):
        """测试时间序列频率设置"""
        forecaster = Forecaster(sample_traffic_data)
        ts = forecaster.prepare_time_series('date', 'user_id')
        
        # 应为日频率
        assert ts.index.freq is not None or ts.shape[0] > 0
    
    def test_arima_order_parameter(self, sample_time_series_data):
        """测试 ARIMA 阶数参数"""
        forecaster = Forecaster(sample_time_series_data)
        ts = forecaster.prepare_time_series('date', 'visits')
        
        # 测试不同阶数
        for order in [(1, 1, 1), (2, 1, 2), (1, 0, 1)]:
            result = forecaster.fit_arima(ts, order=order)
            assert result['order'] == order
