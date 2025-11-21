"""
TSFactory - A Time Series Analysis Framework

一个时序模型的整体框架，包含各类型的时序分析模型。
支持时序数据的统计分析、假设检验、模型辨识、缺失填充、异常检测、序列预测。
支持单维度和多维度输入/输出。
"""

__version__ = "0.1.0"

from .core.data_loader import TimeSeriesLoader
from .core.statistical_analysis import StatisticalAnalyzer
from .core.hypothesis_testing import HypothesisTester
from .core.model_identification import ModelIdentifier
from .models.missing_imputation import MissingValueImputer
from .models.anomaly_detection import AnomalyDetector
from .models.sequence_prediction import SequencePredictor

__all__ = [
    "TimeSeriesLoader",
    "StatisticalAnalyzer",
    "HypothesisTester",
    "ModelIdentifier",
    "MissingValueImputer",
    "AnomalyDetector",
    "SequencePredictor",
]
