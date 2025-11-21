"""
Complete Example: Time Series Analysis with TSFactory

完整示例：使用TSFactory进行时序分析
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# 导入tsfactory模块
from tsfactory import (
    TimeSeriesLoader,
    StatisticalAnalyzer,
    HypothesisTester,
    ModelIdentifier,
    MissingValueImputer,
    AnomalyDetector,
    SequencePredictor
)

def generate_sample_data(n=200, add_trend=True, add_seasonality=True, add_noise=True):
    """
    生成示例时序数据
    """
    t = np.arange(n)
    data = np.zeros(n)
    
    # 趋势
    if add_trend:
        data += 0.05 * t
    
    # 季节性
    if add_seasonality:
        data += 10 * np.sin(2 * np.pi * t / 12)
    
    # 噪声
    if add_noise:
        data += np.random.randn(n) * 2
    
    return data

def main():
    print("=" * 60)
    print("TSFactory Complete Example - 完整示例")
    print("=" * 60)
    
    # 1. 生成和加载数据
    print("\n1. Generating and Loading Data - 生成和加载数据")
    print("-" * 60)
    
    data = generate_sample_data(n=200)
    loader = TimeSeriesLoader()
    loader.load_from_array(data)
    
    print(f"Data shape: {loader.data.shape}")
    print(f"Number of samples: {loader.n_samples}")
    print(f"Number of features: {loader.n_features}")
    
    # 2. 统计分析
    print("\n2. Statistical Analysis - 统计分析")
    print("-" * 60)
    
    analyzer = StatisticalAnalyzer()
    
    # 描述性统计
    stats = analyzer.descriptive_statistics(data)
    print(f"Mean: {stats['mean'][0]:.4f}")
    print(f"Std: {stats['std'][0]:.4f}")
    print(f"Min: {stats['min'][0]:.4f}")
    print(f"Max: {stats['max'][0]:.4f}")
    print(f"Skewness: {stats['skewness'][0]:.4f}")
    print(f"Kurtosis: {stats['kurtosis'][0]:.4f}")
    
    # 自相关分析
    acf = analyzer.autocorrelation(data, max_lag=20)
    print(f"\nACF at lag 1: {acf[1]:.4f}")
    print(f"ACF at lag 12: {acf[12]:.4f}")
    
    # 趋势分析
    trend = analyzer.trend_analysis(data)
    print(f"\nTrend slope: {trend['slope'][0]:.4f}")
    print(f"Trend R^2: {trend['r_squared'][0]:.4f}")
    
    # 季节性强度
    seasonality = analyzer.seasonality_strength(data, period=12)
    print(f"Seasonality strength: {seasonality:.4f}")
    
    # 3. 假设检验
    print("\n3. Hypothesis Testing - 假设检验")
    print("-" * 60)
    
    tester = HypothesisTester()
    
    # 平稳性检验
    adf_result = tester.stationarity_test(data, method='adf')
    print(f"ADF test statistic: {adf_result['test_statistic']:.4f}")
    print(f"Is stationary (ADF): {adf_result['is_stationary']}")
    
    # 正态性检验
    norm_result = tester.normality_test(data)
    print(f"\nNormality test p-value: {norm_result['p_value']:.4f}")
    print(f"Is normal: {norm_result['is_normal']}")
    
    # 白噪声检验
    wn_result = tester.white_noise_test(data, lags=10)
    print(f"\nWhite noise test p-value: {wn_result['p_value']:.4f}")
    print(f"Is white noise: {wn_result['is_white_noise']}")
    
    # 4. 模型辨识
    print("\n4. Model Identification - 模型辨识")
    print("-" * 60)
    
    identifier = ModelIdentifier()
    
    # 识别ARIMA阶数
    arima_order = identifier.identify_arima_order(data, max_p=5, max_d=2, max_q=5)
    print(f"Optimal ARIMA order (p, d, q): {arima_order['order']}")
    
    # 识别AR阶数
    ar_result = identifier.identify_ar_order(data, max_order=10)
    print(f"Optimal AR order: {ar_result['order']}")
    
    # 5. 缺失值填充
    print("\n5. Missing Value Imputation - 缺失值填充")
    print("-" * 60)
    
    # 创建带缺失值的数据
    data_with_missing = data.copy()
    missing_indices = np.random.choice(len(data), size=20, replace=False)
    data_with_missing[missing_indices] = np.nan
    
    imputer = MissingValueImputer(method='linear')
    
    # 缺失值统计
    missing_stats = imputer.missing_statistics(data_with_missing)
    print(f"Total missing values: {missing_stats['total_missing']}")
    print(f"Missing ratio: {missing_stats['missing_ratio']:.4f}")
    
    # 填充缺失值
    filled_data = imputer.fit_transform(data_with_missing)
    print(f"Missing values after imputation: {np.sum(np.isnan(filled_data))}")
    
    # 6. 异常检测
    print("\n6. Anomaly Detection - 异常检测")
    print("-" * 60)
    
    # 添加一些异常点
    data_with_anomalies = data.copy()
    anomaly_indices = [50, 100, 150]
    data_with_anomalies[anomaly_indices] += np.array([20, -25, 30])
    
    # Z-score方法
    detector_zscore = AnomalyDetector(method='zscore', threshold=3.0)
    anomalies_zscore = detector_zscore.detect(data_with_anomalies)
    stats_zscore = detector_zscore.anomaly_statistics(data_with_anomalies)
    print(f"Z-score method - Total anomalies: {stats_zscore['total_anomalies']}")
    
    # IQR方法
    detector_iqr = AnomalyDetector(method='iqr', threshold=1.5)
    stats_iqr = detector_iqr.anomaly_statistics(data_with_anomalies)
    print(f"IQR method - Total anomalies: {stats_iqr['total_anomalies']}")
    
    # 移动平均方法
    detector_ma = AnomalyDetector(method='moving_average', threshold=3.0)
    stats_ma = detector_ma.anomaly_statistics(data_with_anomalies)
    print(f"Moving average method - Total anomalies: {stats_ma['total_anomalies']}")
    
    # 7. 序列预测
    print("\n7. Sequence Prediction - 序列预测")
    print("-" * 60)
    
    # 分割训练集和测试集
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # AR模型
    print("\nAR Model:")
    predictor_ar = SequencePredictor(model_type='ar', order=3)
    predictor_ar.fit(train_data)
    
    # 单步预测评估
    test_data_2d = test_data.reshape(-1, 1)
    X_test, y_test = predictor_ar._prepare_autoregressive_data(test_data_2d, 3)
    metrics_ar = predictor_ar.evaluate(X_test, y_test)
    print(f"  RMSE: {metrics_ar['rmse']:.4f}")
    print(f"  MAE: {metrics_ar['mae']:.4f}")
    print(f"  R^2: {metrics_ar['r2']:.4f}")
    
    # 多步预测
    multi_step_predictions = predictor_ar.predict(
        last_values=train_data[-3:], 
        steps=10
    )
    print(f"  Multi-step predictions (10 steps): {multi_step_predictions[:5]}")
    
    # 线性回归模型
    print("\nLinear Regression Model:")
    predictor_linear = SequencePredictor(model_type='linear')
    
    # 使用时间作为特征
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    predictor_linear.fit(X_train, train_data)
    
    X_test_linear = np.arange(len(train_data), len(data)).reshape(-1, 1)
    metrics_linear = predictor_linear.evaluate(X_test_linear, test_data)
    print(f"  RMSE: {metrics_linear['rmse']:.4f}")
    print(f"  MAE: {metrics_linear['mae']:.4f}")
    print(f"  R^2: {metrics_linear['r2']:.4f}")
    
    # 8. 多维数据示例
    print("\n8. Multi-dimensional Data Example - 多维数据示例")
    print("-" * 60)
    
    # 生成多维数据
    multi_data = np.column_stack([
        generate_sample_data(n=200),
        generate_sample_data(n=200),
        generate_sample_data(n=200)
    ])
    
    loader_multi = TimeSeriesLoader()
    loader_multi.load_from_array(
        multi_data, 
        feature_names=['feature_1', 'feature_2', 'feature_3']
    )
    
    print(f"Multi-dimensional data shape: {loader_multi.data.shape}")
    
    # 多维统计分析
    stats_multi = analyzer.descriptive_statistics(multi_data)
    print(f"Mean per feature: {stats_multi['mean']}")
    print(f"Std per feature: {stats_multi['std']}")
    
    # 多维异常检测
    detector_multi = AnomalyDetector(method='zscore', threshold=3.0)
    stats_multi_anomaly = detector_multi.anomaly_statistics(multi_data)
    print(f"Anomalies per feature: {stats_multi_anomaly['anomalies_per_feature']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully! - 示例运行成功！")
    print("=" * 60)

if __name__ == "__main__":
    main()
