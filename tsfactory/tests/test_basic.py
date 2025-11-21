"""
Basic tests for TSFactory modules

基本测试
"""

import numpy as np
import pytest
from tsfactory import (
    TimeSeriesLoader,
    StatisticalAnalyzer,
    HypothesisTester,
    ModelIdentifier,
    MissingValueImputer,
    AnomalyDetector,
    SequencePredictor
)


class TestTimeSeriesLoader:
    """测试数据加载器"""
    
    def test_load_from_array_1d(self):
        data = np.random.randn(100)
        loader = TimeSeriesLoader()
        loader.load_from_array(data)
        
        assert loader.n_samples == 100
        assert loader.n_features == 1
        assert loader.data.shape == (100, 1)
    
    def test_load_from_array_2d(self):
        data = np.random.randn(100, 3)
        loader = TimeSeriesLoader()
        loader.load_from_array(data)
        
        assert loader.n_samples == 100
        assert loader.n_features == 3
        assert loader.data.shape == (100, 3)
    
    def test_split_train_test(self):
        data = np.random.randn(100, 2)
        loader = TimeSeriesLoader()
        loader.load_from_array(data)
        
        train, test = loader.split_train_test(test_size=0.2)
        assert train.shape[0] == 80
        assert test.shape[0] == 20


class TestStatisticalAnalyzer:
    """测试统计分析器"""
    
    def test_descriptive_statistics(self):
        data = np.random.randn(100)
        analyzer = StatisticalAnalyzer()
        
        stats = analyzer.descriptive_statistics(data)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'var' in stats
        assert isinstance(stats['mean'], np.ndarray)
    
    def test_autocorrelation(self):
        data = np.random.randn(100)
        analyzer = StatisticalAnalyzer()
        
        acf = analyzer.autocorrelation(data, max_lag=10)
        
        assert len(acf) == 11
        assert acf[0] == 1.0
    
    def test_trend_analysis(self):
        data = np.arange(100) + np.random.randn(100) * 0.1
        analyzer = StatisticalAnalyzer()
        
        trend = analyzer.trend_analysis(data)
        
        assert 'slope' in trend
        assert 'intercept' in trend
        assert 'r_squared' in trend


class TestHypothesisTester:
    """测试假设检验器"""
    
    def test_stationarity_test_adf(self):
        data = np.random.randn(100)
        tester = HypothesisTester()
        
        result = tester.stationarity_test(data, method='adf')
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result
    
    def test_normality_test(self):
        data = np.random.randn(100)
        tester = HypothesisTester()
        
        result = tester.normality_test(data)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_normal' in result
    
    def test_white_noise_test(self):
        data = np.random.randn(100)
        tester = HypothesisTester()
        
        result = tester.white_noise_test(data, lags=10)
        
        assert 'test_statistic' in result
        assert 'p_value' in result


class TestModelIdentifier:
    """测试模型辨识器"""
    
    def test_identify_ar_order(self):
        data = np.random.randn(100)
        identifier = ModelIdentifier()
        
        result = identifier.identify_ar_order(data, max_order=5)
        
        assert 'order' in result
        assert result['order'] >= 0
        assert result['order'] <= 5
    
    def test_identify_arima_order(self):
        data = np.random.randn(100)
        identifier = ModelIdentifier()
        
        result = identifier.identify_arima_order(data, max_p=3, max_d=2, max_q=3)
        
        assert 'p' in result
        assert 'd' in result
        assert 'q' in result
        assert 'order' in result


class TestMissingValueImputer:
    """测试缺失值填充器"""
    
    def test_linear_imputation(self):
        data = np.arange(100, dtype=float)
        data[10:15] = np.nan
        
        imputer = MissingValueImputer(method='linear')
        filled = imputer.fit_transform(data)
        
        assert not np.any(np.isnan(filled))
        assert len(filled) == 100
    
    def test_forward_fill(self):
        data = np.arange(100, dtype=float)
        data[10:15] = np.nan
        
        imputer = MissingValueImputer(method='forward')
        filled = imputer.fit_transform(data)
        
        assert not np.any(np.isnan(filled[10:15]))
    
    def test_missing_statistics(self):
        data = np.arange(100, dtype=float)
        data[10:15] = np.nan
        
        imputer = MissingValueImputer()
        stats = imputer.missing_statistics(data)
        
        assert stats['total_missing'] == 5
        assert abs(stats['missing_ratio'] - 0.05) < 1e-6


class TestAnomalyDetector:
    """测试异常检测器"""
    
    def test_zscore_detection(self):
        data = np.random.randn(100)
        data[50] = 10  # 添加异常点
        
        detector = AnomalyDetector(method='zscore', threshold=3.0)
        anomalies = detector.detect(data)
        
        assert len(anomalies) == 100
        assert anomalies[50] == True
    
    def test_iqr_detection(self):
        data = np.random.randn(100)
        
        detector = AnomalyDetector(method='iqr', threshold=1.5)
        anomalies = detector.detect(data)
        
        assert len(anomalies) == 100
    
    def test_anomaly_statistics(self):
        data = np.random.randn(100)
        data[50] = 10
        
        detector = AnomalyDetector(method='zscore', threshold=3.0)
        stats = detector.anomaly_statistics(data)
        
        assert 'total_anomalies' in stats
        assert 'anomaly_ratio' in stats


class TestSequencePredictor:
    """测试序列预测器"""
    
    def test_ar_predictor_fit(self):
        data = np.random.randn(100)
        
        predictor = SequencePredictor(model_type='ar', order=3)
        predictor.fit(data)
        
        assert predictor.is_fitted
        assert predictor.coef_ is not None
    
    def test_ar_predictor_predict(self):
        data = np.random.randn(100)
        
        predictor = SequencePredictor(model_type='ar', order=3)
        predictor.fit(data)
        
        predictions = predictor.predict(last_values=data[-3:], steps=10)
        
        assert len(predictions) == 10
    
    def test_linear_predictor(self):
        X = np.arange(100).reshape(-1, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.1
        
        predictor = SequencePredictor(model_type='linear')
        predictor.fit(X, y)
        
        predictions = predictor.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_evaluate(self):
        data = np.random.randn(100)
        
        predictor = SequencePredictor(model_type='ar', order=3)
        predictor.fit(data[:80])
        
        test_data = data[80:].reshape(-1, 1)
        X_test, y_test = predictor._prepare_autoregressive_data(test_data, 3)
        metrics = predictor.evaluate(X_test, y_test)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
