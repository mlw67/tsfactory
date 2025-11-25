"""
Deep Learning Models

深度学习模型，基于base_models构建，包含不同任务的预测头

功能模型包括:
- 预测模型 (Forecasting): 短期参数预测
- 异常检测 (Anomaly Detection): 点分类、区间分类
- 故障诊断 (Fault Diagnosis): 样本分类

所有功能模型支持:
- 模型权重加载 (load_from_dict, load_from_file)
- Head 部分微调 (fine_tune_head)
- 整体权重重新训练 (train_full)
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

# Training utilities
from .training import (
    ClassifierTrainer,
    RegressionTrainer,
    TrainingMixin,
    create_training_data,
    initialize_feature_extractor
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
    # Training utilities
    "ClassifierTrainer",
    "RegressionTrainer",
    "TrainingMixin",
    "create_training_data",
    "initialize_feature_extractor",
]
