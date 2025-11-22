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
│   │   │   └── positional_encoding.py  # 位置编码 / Positional encoding
│   │   ├── dlinear.py            # DLinear基础模型 / DLinear base model
│   │   └── transformer.py        # Transformer基础模型 / Transformer base model
│   │
│   ├── statistical/              # 统计模型 / Statistical models
│   │   ├── anomaly_detection.py  # 异常检测 / Anomaly detection
│   │   └── missing_imputation.py # 缺失值填充 / Missing value imputation
│   │
│   ├── machine_learning/         # 机器学习模型 / Machine learning models
│   │   └── sequence_prediction.py # 序列预测 / Sequence prediction
│   │
│   └── deep_learning/            # 深度学习模型 / Deep learning models
│       └── forecasting.py        # 预测模型 / Forecasting models
│           ├── DLinearForecaster
│           └── TransformerForecaster
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

#### TransformerBaseModel
- 编码器-解码器架构 / Encoder-decoder architecture
- 多头注意力机制 / Multi-head attention
- 位置编码 / Positional encoding
- 使用 PositionalEncoding 层 / Uses PositionalEncoding layer

### 3. 共享层 (Shared Layers)

共享层位于 `models/base_models/layers/`，被多个基础模型使用：

Shared layers in `models/base_models/layers/` are used by multiple base models:

- `SeriesDecomp`: 时序分解层，使用移动平均提取趋势 / Decomposes time series using moving average
- `PositionalEncoding`: 为 Transformer 生成位置编码 / Generates positional encodings for Transformers

### 4. 任务特定模型 (Task-Specific Models)

#### 深度学习模型 (Deep Learning Models)

位于 `models/deep_learning/`，继承自基础模型并添加任务特定的功能：

In `models/deep_learning/`, inherit from base models and add task-specific functionality:

- `DLinearForecaster`: 基于 DLinearBaseModel 的预测模型 / Forecasting model based on DLinearBaseModel
- `TransformerForecaster`: 基于 TransformerBaseModel 的预测模型 / Forecasting model based on TransformerBaseModel

这些模型：
These models:
- 继承基础模型的架构 / Inherit base model architecture
- 添加预测任务的头部 / Add task-specific heads for forecasting
- 提供简化的预测接口 / Provide simplified prediction interface

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
from tsfactory.models.base_models.layers import SeriesDecomp, PositionalEncoding
import numpy as np

# 时序分解 / Series decomposition
decomp = SeriesDecomp(kernel_size=25)
x = np.random.randn(1, 100, 1)
seasonal, trend = decomp(x)

# 位置编码 / Positional encoding
pe = PositionalEncoding(d_model=512)
x = np.random.randn(1, 100, 512)
x_with_pe = pe(x)
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
   - 例如: Informer, Autoformer, FEDformer 等

2. **新的任务** / **New Tasks**:
   - 在 `models/deep_learning/` 中添加新的任务特定模型
   - 例如: 分类 (Classification), 回归 (Regression), 异常检测 (Anomaly Detection)

3. **新的共享层** / **New Shared Layers**:
   - 在 `models/base_models/layers/` 中添加新的层
   - 例如: Attention layers, Normalization layers 等
