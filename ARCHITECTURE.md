# TSFactory Architecture Documentation

## 项目结构 / Project Structure

TSFactory 现在采用模块化的架构，将深度学习相关组件组织如下：

TSFactory now uses a modular architecture with deep learning components organized as follows:

```
tsfactory/
├── core/                          # 核心模块 / Core modules
│   ├── data_loader.py            # 时序数据加载 / Time series data loader
│   ├── data_processor.py         # 数据处理 / Data processing
│   ├── dl_data_loader.py         # 深度学习模型权重加载器 / DL model weight loaders
│   ├── statistical_analysis.py   # 统计分析 / Statistical analysis
│   ├── hypothesis_testing.py     # 假设检验 / Hypothesis testing
│   └── model_identification.py   # 模型辨识 / Model identification
│
├── models/                        # 模型目录 / Models directory
│   ├── base_models/              # 基础模型 / Base models
│   │   ├── layers/               # 共享层 / Shared layers
│   │   │   ├── series_decomp.py  # 时序分解层 / Series decomposition
│   │   │   ├── positional_encoding.py  # 位置编码 / Positional encoding
│   │   │   ├── autocorrelation.py      # 自相关层 / Autocorrelation
│   │   │   ├── embedding.py            # 嵌入层 / Embedding layers
│   │   │   ├── attention.py            # 注意力层 / Attention layers
│   │   │   ├── feedforward.py          # 前馈网络层 / Feed-forward layers
│   │   │   └── normalization.py        # 归一化层 / Normalization layers
│   │   ├── dlinear.py            # DLinear基础模型 / DLinear base model
│   │   ├── nlinear.py            # NLinear基础模型 / NLinear base model
│   │   ├── transformer.py        # Transformer基础模型 / Transformer base model
│   │   ├── autoformer.py         # Autoformer基础模型 / Autoformer base model
│   │   ├── timexer.py            # TimeXer基础模型 / TimeXer base model
│   │   └── timemixer.py          # TimeMixer基础模型 / TimeMixer base model
│   │
│   ├── statistical/              # 统计模型 / Statistical models
│   │   ├── anomaly_detection.py  # 异常检测 / Anomaly detection
│   │   └── missing_imputation.py # 缺失值填充 / Missing value imputation
│   │
│   ├── machine_learning/         # 机器学习模型 / Machine learning models
│   │   └── sequence_prediction.py # 序列预测 / Sequence prediction
│   │
│   └── deep_learning/            # 深度学习模型 / Deep learning models
│       ├── forecasting.py        # 预测模型 / Forecasting models
│       │   ├── DLinearForecaster
│       │   └── TransformerForecaster
│       ├── parameter_prediction.py  # 参数预测（短期预测）/ Parameter prediction
│       │   ├── NLinearForecaster
│       │   ├── AutoformerForecaster
│       │   ├── TimeXerForecaster
│       │   ├── TimeMixerForecaster
│       │   └── ShortTermPredictor
│       ├── anomaly_detection.py  # 深度学习异常检测 / DL Anomaly detection
│       │   ├── PointAnomalyDetector    # 点分类 / Point classification
│       │   └── IntervalAnomalyDetector # 区间分类 / Interval classification
│       └── fault_diagnosis.py    # 故障诊断（样本分类）/ Fault diagnosis
│           ├── FaultDiagnosisClassifier
│           └── MultiFaultDiagnosisClassifier
│
└── tests/                        # 测试 / Tests
```

## 核心概念 / Core Concepts

### 1. 数据加载器 (Data Loaders)

数据加载器位于 `core/dl_data_loader.py`，负责从文件中加载预训练模型的权重：

Data loaders in `core/dl_data_loader.py` load pre-trained model weights from files:

- `DLinearDataLoader`: 加载 DLinear 模型权重 / Loads DLinear model weights
- `TransformerDataLoader`: 加载 Transformer 模型权重 / Loads Transformer model weights

支持的格式 / Supported formats:
- PyTorch (.pth, .pt)
- NumPy (.npz, .npy)
- Python dictionaries

### 2. 基础模型 (Base Models)

基础模型位于 `models/base_models/`，提供核心的深度学习架构：

Base models in `models/base_models/` provide core deep learning architectures:

#### DLinearBaseModel
- 时序分解（趋势 + 季节性）/ Time series decomposition (trend + seasonal)
- 线性预测层 / Linear prediction layers
- 使用 SeriesDecomp 层 / Uses SeriesDecomp layer

#### NLinearBaseModel
- 最后值归一化方案 / Last-value normalization scheme
- 处理分布偏移 / Handles distribution shift
- 简单线性层 / Simple linear layer

#### TransformerBaseModel
- 编码器-解码器架构 / Encoder-decoder architecture
- 多头注意力机制 / Multi-head attention
- 位置编码 / Positional encoding
- 使用 PositionalEncoding 层 / Uses PositionalEncoding layer

#### AutoformerBaseModel
- 自相关机制 / Auto-correlation mechanism
- 逐步分解架构 / Progressive decomposition
- 编码器-解码器结构 / Encoder-decoder structure
- 使用 AutoCorrelation 和 SeriesDecomp 层

#### TimeXerBaseModel
- Patch嵌入 / Patch embedding
- 外生变量支持 / Exogenous variable support
- 交叉注意力机制 / Cross-attention mechanism
- 使用 PatchEmbedding 和 FullAttention 层

#### TimeMixerBaseModel
- 多尺度混合 / Multi-scale mixing
- 季节-趋势分解 / Seasonal-trend decomposition
- RevIN归一化 / RevIN normalization
- 使用 SeriesDecomp 和 RevIN 层

### 3. 共享层 (Shared Layers)

共享层位于 `models/base_models/layers/`，被多个基础模型使用：

Shared layers in `models/base_models/layers/` are used by multiple base models:

- `SeriesDecomp`: 时序分解层，使用移动平均提取趋势 / Decomposes time series using moving average
- `PositionalEncoding`: 为 Transformer 生成位置编码 / Generates positional encodings for Transformers
- `AutoCorrelation`: 自相关层，用于 Autoformer / Auto-correlation layer for Autoformer
- `TokenEmbedding`: Token嵌入层 / Token embedding layer
- `PositionalEmbedding`: 位置嵌入层 / Positional embedding layer
- `TemporalEmbedding`: 时间嵌入层 / Temporal embedding layer
- `DataEmbedding`: 数据嵌入层 / Data embedding layer
- `PatchEmbedding`: Patch嵌入层 / Patch embedding layer
- `FullAttention`: 全注意力层 / Full attention layer
- `ProbAttention`: 稀疏注意力层 / Sparse attention layer
- `CrossAttention`: 交叉注意力层 / Cross-attention layer
- `FeedForward`: 前馈网络层 / Feed-forward layer
- `LayerNorm`: 层归一化 / Layer normalization
- `RevIN`: 可逆实例归一化 / Reversible instance normalization
- `BatchNorm1d`: 批归一化 / Batch normalization
- `RMSNorm`: RMS归一化 / RMS normalization

### 4. 任务特定模型 (Task-Specific Models)

#### 深度学习模型 (Deep Learning Models)

位于 `models/deep_learning/`，继承自基础模型并添加任务特定的功能：

In `models/deep_learning/`, inherit from base models and add task-specific functionality:

**预测模型 (Forecasting Models) - 参数预测/短期预测:**
- `DLinearForecaster`: 基于 DLinearBaseModel 的预测模型 / Forecasting model based on DLinearBaseModel
- `TransformerForecaster`: 基于 TransformerBaseModel 的预测模型 / Forecasting model based on TransformerBaseModel
- `NLinearForecaster`: 基于 NLinearBaseModel 的预测模型 / Forecasting model based on NLinearBaseModel
- `AutoformerForecaster`: 基于 AutoformerBaseModel 的预测模型 / Forecasting model based on AutoformerBaseModel
- `TimeXerForecaster`: 基于 TimeXerBaseModel 的预测模型 / Forecasting model based on TimeXerBaseModel
- `TimeMixerForecaster`: 基于 TimeMixerBaseModel 的预测模型 / Forecasting model based on TimeMixerBaseModel
- `ShortTermPredictor`: 统一的短期预测接口 / Unified short-term prediction interface

**异常检测模型 (Anomaly Detection Models) - 点分类/区间分类:**
- `PointAnomalyDetector`: 点级别异常检测（点分类）/ Point-level anomaly detection (point classification)
- `IntervalAnomalyDetector`: 区间级别异常检测（区间分类）/ Interval-level anomaly detection (interval classification)

**故障诊断模型 (Fault Diagnosis Models) - 样本分类:**
- `FaultDiagnosisClassifier`: 单标签故障分类 / Single-label fault classification
- `MultiFaultDiagnosisClassifier`: 多标签故障分类 / Multi-label fault classification

这些模型：
These models:
- 继承基础模型的架构 / Inherit base model architecture
- 添加任务特定的头部 / Add task-specific heads
- 提供简化的接口 / Provide simplified interface

#### 统计模型 (Statistical Models)

位于 `models/statistical/`：

In `models/statistical/`:
- `AnomalyDetector`: 异常检测 / Anomaly detection
- `MissingValueImputer`: 缺失值填充 / Missing value imputation

#### 机器学习模型 (Machine Learning Models)

位于 `models/machine_learning/`：

In `models/machine_learning/`:
- `SequencePredictor`: AR、线性回归、多项式回归 / AR, linear regression, polynomial regression

## 使用示例 / Usage Examples

### 基本用法 / Basic Usage

```python
from tsfactory import DLinearForecaster, TransformerForecaster
from tsfactory import DLinearDataLoader, TransformerDataLoader
import numpy as np

# 1. 创建预测模型 / Create forecaster
dlinear = DLinearForecaster(seq_len=96, pred_len=24, enc_in=1)

# 2. 使用数据加载器加载权重 / Load weights using data loader
# 方式1: 从文件加载 / Method 1: Load from file
dlinear.load_from_file('model.pth')

# 方式2: 从字典加载 / Method 2: Load from dictionary
weights = DLinearDataLoader.load_from_file('model.pth')
dlinear.load_from_dict(weights)

# 3. 进行预测 / Make predictions
input_data = np.random.randn(96)
predictions = dlinear.predict(input_data)
```

### 高级用法：使用基础模型 / Advanced Usage: Using Base Models

```python
from tsfactory.models.base_models import DLinearBaseModel
from tsfactory.core import DLinearDataLoader

# 1. 创建基础模型 / Create base model
base_model = DLinearBaseModel(seq_len=96, pred_len=24, enc_in=1)

# 2. 加载权重 / Load weights
weights = DLinearDataLoader.load_from_file('model.pth')
base_model.set_weights(
    seasonal_weights=weights.get('seasonal'),
    trend_weights=weights.get('trend')
)

# 3. 前向传播 / Forward pass
input_data = np.random.randn(1, 96, 1)  # (batch, seq_len, features)
output = base_model.forward(input_data)
```

### 使用共享层 / Using Shared Layers

```python
from tsfactory.models.base_models.layers import (
    SeriesDecomp, PositionalEncoding, AutoCorrelation,
    TokenEmbedding, PatchEmbedding, FullAttention, 
    RevIN, LayerNorm, FeedForward
)
import numpy as np

# 时序分解 / Series decomposition
decomp = SeriesDecomp(kernel_size=25)
x = np.random.randn(1, 100, 1)
seasonal, trend = decomp(x)

# 位置编码 / Positional encoding
pe = PositionalEncoding(d_model=512)
x = np.random.randn(1, 100, 512)
x_with_pe = pe(x)

# RevIN 归一化 / RevIN normalization
revin = RevIN(num_features=7)
x = np.random.randn(1, 96, 7)
x_norm = revin(x, mode='norm')
x_denorm = revin(x_norm, mode='denorm')

# Patch 嵌入 / Patch embedding
patch_embed = PatchEmbedding(d_model=256, patch_len=16, stride=8)
x = np.random.randn(1, 96, 7)
patches = patch_embed(x)
```

### 使用异常检测模型 / Using Anomaly Detection Models

```python
from tsfactory import PointAnomalyDetector, IntervalAnomalyDetector
import numpy as np

# 点异常检测 / Point anomaly detection
detector = PointAnomalyDetector(seq_len=96, n_features=1, threshold=0.5)
detector.load_from_dict({'encoder': {}, 'classifier': {}})

x = np.random.randn(96)
labels, scores = detector.detect(x)

# 区间异常检测 / Interval anomaly detection
interval_detector = IntervalAnomalyDetector(
    seq_len=96, interval_len=16, n_features=1, threshold=0.5
)
interval_detector.load_from_dict({'feature_extractor': {}, 'classifier': {}})

x = np.random.randn(96)
labels, scores, positions = interval_detector.detect(x)
```

### 使用故障诊断模型 / Using Fault Diagnosis Models

```python
from tsfactory import FaultDiagnosisClassifier, MultiFaultDiagnosisClassifier
import numpy as np

# 故障分类 / Fault classification
classifier = FaultDiagnosisClassifier(seq_len=96, n_classes=3, n_features=1)
classifier.set_class_names(['Normal', 'Fault_A', 'Fault_B'])
classifier.load_from_dict({'feature_extractor': {}, 'classifier': {}})

x = np.random.randn(96)
predictions, probabilities = classifier.predict(x)

# 多故障诊断 / Multi-fault diagnosis
multi_classifier = MultiFaultDiagnosisClassifier(
    seq_len=96, n_faults=3, n_features=1, threshold=0.5
)
multi_classifier.set_fault_names(['Overheating', 'Vibration', 'Leakage'])
multi_classifier.load_from_dict({'feature_extractor': {}, 'classifiers': {}})

x = np.random.randn(96)
predictions, probabilities = multi_classifier.predict(x)
```

### 使用短期预测模型 / Using Short-term Prediction Models

```python
from tsfactory import ShortTermPredictor, NLinearForecaster
import numpy as np

# 统一接口 / Unified interface
predictor = ShortTermPredictor(
    seq_len=96, pred_len=24, model_type='nlinear'
)
# predictor.load_from_file('model.pth')

# 或使用特定模型 / Or use specific model
nlinear = NLinearForecaster(seq_len=96, pred_len=24, enc_in=1)
# nlinear.load_from_file('nlinear_model.pth')

# 预测 / Prediction
x = np.random.randn(96)
# predictions = predictor.predict(x)
```

## 向后兼容性 / Backward Compatibility

为了保持向后兼容性，提供了别名：

For backward compatibility, aliases are provided:

```python
# 旧的导入方式仍然有效 / Old imports still work
from tsfactory import DLinearLoader, TransformerLoader

# 这些是新类的别名 / These are aliases for new classes
# DLinearLoader = DLinearForecaster
# TransformerLoader = TransformerForecaster
```

## 设计原则 / Design Principles

1. **模块化** / **Modularity**: 每个组件有明确的职责
   - 数据加载器：加载权重 / Data loaders: Load weights
   - 基础模型：提供核心架构 / Base models: Provide core architecture
   - 任务模型：添加特定功能 / Task models: Add specific functionality

2. **可复用性** / **Reusability**: 共享层可被多个模型使用
   - SeriesDecomp 用于所有需要时序分解的模型
   - PositionalEncoding 用于所有 Transformer 类模型

3. **可扩展性** / **Extensibility**: 容易添加新的模型和任务
   - 继承基础模型创建新的任务特定模型
   - 添加新的共享层供多个模型使用

4. **清晰的分层** / **Clear Hierarchy**:
   - Core: 数据加载和处理 / Data loading and processing
   - Base Models: 深度学习架构 / Deep learning architectures  
   - Task Models: 特定任务实现 / Task-specific implementations

## 测试 / Testing

所有组件都有对应的测试：
All components have corresponding tests:

```bash
# 运行所有测试 / Run all tests
pytest tsfactory/tests/

# 运行深度学习测试 / Run deep learning tests
pytest tsfactory/tests/test_deep_learning.py

# 运行特定测试 / Run specific tests
pytest tsfactory/tests/test_deep_learning.py::TestDLinearLoader
```

## 未来扩展 / Future Extensions

可以轻松添加新功能：
New features can be easily added:

1. **新的基础模型** / **New Base Models**:
   - 在 `models/base_models/` 中添加新的架构
   - 例如: Informer, FEDformer, PatchTST 等

2. **新的任务** / **New Tasks**:
   - 在 `models/deep_learning/` 中添加新的任务特定模型
   - 例如: 分类 (Classification), 回归 (Regression), 异常检测 (Anomaly Detection)

3. **新的共享层** / **New Shared Layers**:
   - 在 `models/base_models/layers/` 中添加新的层
   - 例如: 新的注意力机制, 新的归一化层 等

## 深度学习功能模型分类 / Deep Learning Task Models

| 任务类型 | 模型 | 说明 |
|---------|------|------|
| 异常检测-点分类 | PointAnomalyDetector | 检测单个时间点的异常 |
| 异常检测-区间分类 | IntervalAnomalyDetector | 检测时间区间的异常 |
| 故障诊断-样本分类 | FaultDiagnosisClassifier | 单标签故障分类 |
| 故障诊断-样本分类 | MultiFaultDiagnosisClassifier | 多标签故障分类 |
| 参数预测-短期预测 | DLinearForecaster | DLinear预测 |
| 参数预测-短期预测 | NLinearForecaster | NLinear预测 |
| 参数预测-短期预测 | TransformerForecaster | Transformer预测 |
| 参数预测-短期预测 | AutoformerForecaster | Autoformer预测 |
| 参数预测-短期预测 | TimeXerForecaster | TimeXer预测 |
| 参数预测-短期预测 | TimeMixerForecaster | TimeMixer预测 |
| 参数预测-短期预测 | ShortTermPredictor | 统一预测接口 |
