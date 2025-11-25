"""
TSFactory - A Time Series Analysis Framework

一个时序模型的整体框架，包含各类型的时序分析模型。
支持时序数据的统计分析、假设检验、模型辨识、缺失填充、异常检测、序列预测。
支持单维度和多维度输入/输出。

深度学习功能模型包括:
- 异常检测（点分类、区间分类）
- 故障诊断（样本分类）
- 参数预测（短期预测）
"""

__version__ = "0.1.0"

# Core modules
from .core.data_loader import TimeSeriesLoader
from .core.data_processor import DataProcessor
from .core.statistical_analysis import StatisticalAnalyzer
from .core.hypothesis_testing import HypothesisTester
from .core.model_identification import ModelIdentifier
from .core.dl_data_loader import DLinearDataLoader, TransformerDataLoader

# Statistical models
from .models.statistical import MissingValueImputer, AnomalyDetector

# Machine learning models
from .models.machine_learning import SequencePredictor

# Deep learning models - Forecasting
from .models.deep_learning import (
    DLinearForecaster,
    TransformerForecaster,
    NLinearForecaster,
    AutoformerForecaster,
    TimeXerForecaster,
    TimeMixerForecaster,
    ShortTermPredictor
)

# Deep learning models - Anomaly Detection
from .models.deep_learning import (
    PointAnomalyDetector,
    IntervalAnomalyDetector
)

# Deep learning models - Fault Diagnosis
from .models.deep_learning import (
    FaultDiagnosisClassifier,
    MultiFaultDiagnosisClassifier
)

# For backward compatibility
DLinearLoader = DLinearForecaster
TransformerLoader = TransformerForecaster

__all__ = [
    # Core
    "TimeSeriesLoader",
    "DataProcessor",
    "StatisticalAnalyzer",
    "HypothesisTester",
    "ModelIdentifier",
    "DLinearDataLoader",
    "TransformerDataLoader",
    # Statistical models
    "MissingValueImputer",
    "AnomalyDetector",
    # Machine learning models
    "SequencePredictor",
    # Deep learning forecasting models
    "DLinearForecaster",
    "TransformerForecaster",
    "NLinearForecaster",
    "AutoformerForecaster",
    "TimeXerForecaster",
    "TimeMixerForecaster",
    "ShortTermPredictor",
    # Deep learning anomaly detection
    "PointAnomalyDetector",
    "IntervalAnomalyDetector",
    # Deep learning fault diagnosis
    "FaultDiagnosisClassifier",
    "MultiFaultDiagnosisClassifier",
    # Backward compatibility aliases
    "DLinearLoader",
    "TransformerLoader",
]
