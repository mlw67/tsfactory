"""
Models for Time Series Analysis

时序分析模型，包括统计模型、机器学习模型和深度学习模型
"""

# Statistical models
from .statistical import AnomalyDetector, MissingValueImputer

# Machine learning models  
from .machine_learning import SequencePredictor

# Deep learning models - Forecasting
from .deep_learning import (
    DLinearForecaster,
    TransformerForecaster,
    NLinearForecaster,
    AutoformerForecaster,
    TimeXerForecaster,
    TimeMixerForecaster,
    ShortTermPredictor
)

# Deep learning models - Anomaly Detection
from .deep_learning import (
    PointAnomalyDetector,
    IntervalAnomalyDetector
)

# Deep learning models - Fault Diagnosis
from .deep_learning import (
    FaultDiagnosisClassifier,
    MultiFaultDiagnosisClassifier
)

# Base models (for advanced users)
from .base_models import (
    DLinearBaseModel,
    NLinearBaseModel,
    TransformerBaseModel,
    AutoformerBaseModel,
    TimeXerBaseModel,
    TimeMixerBaseModel
)

__all__ = [
    # Statistical models
    "AnomalyDetector",
    "MissingValueImputer",
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
    # Base models
    "DLinearBaseModel",
    "NLinearBaseModel",
    "TransformerBaseModel",
    "AutoformerBaseModel",
    "TimeXerBaseModel",
    "TimeMixerBaseModel",
]
