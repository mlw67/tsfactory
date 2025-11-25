"""
Deep Learning Models

深度学习模型，基于base_models构建，包含不同任务的预测头

功能模型包括:
- 预测模型 (Forecasting): 短期参数预测
- 异常检测 (Anomaly Detection): 点分类、区间分类
- 故障诊断 (Fault Diagnosis): 样本分类
"""

# Forecasting models
from .forecasting import DLinearForecaster, TransformerForecaster

# Parameter prediction models
from .parameter_prediction import (
    NLinearForecaster,
    AutoformerForecaster,
    TimeXerForecaster,
    TimeMixerForecaster,
    ShortTermPredictor
)

# Anomaly detection models
from .anomaly_detection import (
    PointAnomalyDetector,
    IntervalAnomalyDetector
)

# Fault diagnosis models
from .fault_diagnosis import (
    FaultDiagnosisClassifier,
    MultiFaultDiagnosisClassifier
)

__all__ = [
    # Forecasting models
    "DLinearForecaster",
    "TransformerForecaster",
    # Parameter prediction
    "NLinearForecaster",
    "AutoformerForecaster",
    "TimeXerForecaster",
    "TimeMixerForecaster",
    "ShortTermPredictor",
    # Anomaly detection
    "PointAnomalyDetector",
    "IntervalAnomalyDetector",
    # Fault diagnosis
    "FaultDiagnosisClassifier",
    "MultiFaultDiagnosisClassifier",
]
