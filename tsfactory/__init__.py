"""
TSFactory - A Time Series Analysis Framework

一个时序模型的整体框架，包含各类型的时序分析模型。
支持时序数据的统计分析、假设检验、模型辨识、缺失填充、异常检测、序列预测。
支持单维度和多维度输入/输出。
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

# Deep learning models
from .models.deep_learning import DLinearForecaster, TransformerForecaster

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
    # Deep learning models
    "DLinearForecaster",
    "TransformerForecaster",
    # Backward compatibility aliases
    "DLinearLoader",
    "TransformerLoader",
]
