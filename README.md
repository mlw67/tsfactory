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
- **深度学习模型 / Deep Learning Models**: DLinear、Transformer编码解码器等深度学习模型加载器
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

# 时序转图像 (统一的3D格式，单特征时n_features=1)
image = processor.to_image(data, period=24)  # shape: (n_rows, 24, 1) or (n_rows, 24, n_features)

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

### 9. 深度学习模型 / Deep Learning Models

```python
from tsfactory import DLinearLoader, TransformerLoader
import numpy as np

# DLinear模型加载和使用
dlinear = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)

# 从文件加载预训练模型
# dlinear.load_from_file('dlinear_model.pth')

# 或者从字典加载模型权重
state_dict = {
    'seasonal_Linear.weight': np.random.randn(24, 96),
    'seasonal_Linear.bias': np.random.randn(24),
    'trend_Linear.weight': np.random.randn(24, 96),
    'trend_Linear.bias': np.random.randn(24)
}
dlinear.load_from_dict(state_dict)

# 进行预测
input_data = np.random.randn(96)  # 输入序列
predictions = dlinear.predict(input_data)  # 预测未来24步
print(f"DLinear predictions shape: {predictions.shape}")

# Transformer模型加载和使用
transformer = TransformerLoader(
    seq_len=96, 
    pred_len=24, 
    d_model=512, 
    n_heads=8
)

# 从文件加载预训练模型
# transformer.load_from_file('transformer_model.pth')

# 或者从字典加载模型权重
state_dict = {
    'encoder.layer0.weight': np.random.randn(512, 512),
    'decoder.layer0.weight': np.random.randn(512, 512),
    'projection.weight': np.random.randn(24, 96),
}
transformer.load_from_dict(state_dict)

# 进行预测
predictions = transformer.predict(input_data)
print(f"Transformer predictions shape: {predictions.shape}")

# 查看模型信息
dlinear_info = dlinear.get_model_info()
print(f"DLinear model: {dlinear_info['model_type']}, seq_len={dlinear_info['seq_len']}")

transformer_info = transformer.get_model_info()
print(f"Transformer model: {transformer_info['model_type']}, d_model={transformer_info['d_model']}")
```
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
