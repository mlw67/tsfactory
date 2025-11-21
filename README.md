# TSFactory

一个时序模型的整体框架 / A Time Series Analysis Framework

## 简介 / Introduction

TSFactory 是一个综合性的时序分析框架，包含各类型的时序分析模型。支持时序数据的统计分析、假设检验、模型辨识、缺失填充、异常检测、序列预测等功能。

TSFactory is a comprehensive time series analysis framework that includes various types of time series analysis models. It supports statistical analysis, hypothesis testing, model identification, missing value imputation, anomaly detection, and sequence prediction.

## 主要特性 / Key Features

- **数据加载 / Data Loading**: 支持从多种数据源加载时序数据
- **数据处理 / Data Processing**: 数据滤波、时序分解、时序转图像
- **统计分析 / Statistical Analysis**: 描述性统计、自相关分析、趋势分析、季节性分析
- **假设检验 / Hypothesis Testing**: 平稳性检验、正态性检验、相关性检验、白噪声检验
- **模型辨识 / Model Identification**: ARIMA模型阶数识别
- **缺失值填充 / Missing Value Imputation**: 多种插值方法（线性、样条、前向/后向填充等）
- **异常检测 / Anomaly Detection**: Z-score、IQR、孤立森林、移动平均等方法
- **序列预测 / Sequence Prediction**: AR、线性回归、多项式回归等模型
- **多维支持 / Multi-dimensional Support**: 支持单维度和多维度输入/输出

## 安装 / Installation

```bash
pip install -r requirements.txt
```

或者从源码安装 / Or install from source:

```bash
git clone https://github.com/mlw67/tsfactory.git
cd tsfactory
pip install -e .
```

## 快速开始 / Quick Start

### 1. 数据加载 / Data Loading

```python
import numpy as np
from tsfactory import TimeSeriesLoader

# 从numpy数组加载
data = np.random.randn(100)
loader = TimeSeriesLoader()
loader.load_from_array(data)

# 从CSV文件加载
loader.load_from_csv('data.csv', timestamp_col='date')
```

### 2. 数据处理 / Data Processing

```python
from tsfactory import DataProcessor

processor = DataProcessor()

# 数据滤波
filtered = processor.filter(data, method='moving_average', window_size=5)
filtered = processor.filter(data, method='savgol', window_size=11, polyorder=2)

# 时序分解
components = processor.decompose(data, method='additive', period=12)
trend = components['trend']
seasonal = components['seasonal']
residual = components['residual']

# 时序转图像 (每个周期为一行，多特征为多通道)
image = processor.to_image(data, period=24)  # shape: (n_rows, 24) or (n_rows, 24, n_features)

# 图像转回时序
recovered = processor.from_image(image, norm_min=0, norm_max=1)
```

### 3. 统计分析 / Statistical Analysis

```python
from tsfactory import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# 描述性统计
stats = analyzer.descriptive_statistics(data)
print(f"Mean: {stats['mean']}, Std: {stats['std']}")

# 自相关分析
acf = analyzer.autocorrelation(data, max_lag=20)

# 趋势分析
trend = analyzer.trend_analysis(data)
```

### 4. 假设检验 / Hypothesis Testing

```python
from tsfactory import HypothesisTester

tester = HypothesisTester()

# 平稳性检验
result = tester.stationarity_test(data, method='adf')
print(f"Is stationary: {result['is_stationary']}")

# 正态性检验
result = tester.normality_test(data)
print(f"Is normal: {result['is_normal']}")
```

### 5. 模型辨识 / Model Identification

```python
from tsfactory import ModelIdentifier

identifier = ModelIdentifier()

# 识别ARIMA阶数
result = identifier.identify_arima_order(data, max_p=5, max_d=2, max_q=5)
print(f"Optimal ARIMA order: {result['order']}")
```

### 6. 缺失值填充 / Missing Value Imputation

```python
from tsfactory import MissingValueImputer

# 创建带缺失值的数据
data_with_missing = data.copy()
data_with_missing[10:15] = np.nan

# 线性插值填充
imputer = MissingValueImputer(method='linear')
filled_data = imputer.fit_transform(data_with_missing)
```

### 7. 异常检测 / Anomaly Detection

```python
from tsfactory import AnomalyDetector

# Z-score方法
detector = AnomalyDetector(method='zscore', threshold=3.0)
anomalies = detector.detect(data)

# 获取异常统计
stats = detector.anomaly_statistics(data)
print(f"Total anomalies: {stats['total_anomalies']}")
```

### 8. 序列预测 / Sequence Prediction

```python
from tsfactory import SequencePredictor

# AR模型
predictor = SequencePredictor(model_type='ar', order=3)
predictor.fit(data)

# 多步预测
predictions = predictor.predict(last_values=data[-3:], steps=10)

# 模型评估
train_data, test_data = data[:80], data[80:]
predictor.fit(train_data)
X_test, y_test = predictor._prepare_autoregressive_data(test_data, 3)
metrics = predictor.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']}")
```

## 完整示例 / Complete Example

查看 `examples/` 目录获取更多示例。

See the `examples/` directory for more examples.

## 文档 / Documentation

详细文档请参见各模块的docstring。

For detailed documentation, please refer to the docstrings in each module.

## 依赖 / Dependencies

- numpy >= 1.19.0
- pandas >= 1.1.0
- scipy >= 1.5.0

## 许可 / License

MIT License

## 贡献 / Contributing

欢迎贡献！请提交issue或pull request。

Contributions are welcome! Please submit issues or pull requests.
