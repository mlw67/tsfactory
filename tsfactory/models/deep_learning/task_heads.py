"""
Task-specific Heads for Deep Learning Models

任务特定的预测头，参考 Time-Series-Library 的实现：
- long_term_forecast / short_term_forecast: Linear(d_model, pred_len)
- imputation: Linear(d_model, seq_len)
- anomaly_detection: Linear(d_model, seq_len)
- classification: GELU + Dropout + Linear(d_model * enc_in, num_class)

支持:
- 模型权重加载
- Head 部分微调 (fine_tune_head)
- 整体权重重新训练 (train_full)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List


# Precomputed constant for GELU activation
_GELU_CONST = np.sqrt(2 / np.pi)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU 激活函数
    Gaussian Error Linear Unit activation
    
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(_GELU_CONST * (x + 0.044715 * x ** 3)))


def dropout(x: np.ndarray, p: float = 0.1, training: bool = True) -> np.ndarray:
    """
    Dropout 正则化
    
    Parameters:
    -----------
    x : np.ndarray
        输入张量
    p : float
        dropout 概率
    training : bool
        是否处于训练模式
    """
    if not training or p == 0:
        return x
    mask = np.random.binomial(1, 1 - p, x.shape) / (1 - p)
    return x * mask


class ForecastHead:
    """
    预测头 (长期预测/短期预测)
    
    Forecasting head for long-term and short-term prediction.
    Structure: Linear(d_model, pred_len)
    
    Reference:
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    """
    
    def __init__(self, d_model: int, pred_len: int):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        pred_len : int
            预测长度 / Prediction length
        """
        self.d_model = d_model
        self.pred_len = pred_len
        self.weight = None
        self.bias = None
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        scale = np.sqrt(2.0 / (self.d_model + self.pred_len))
        self.weight = np.random.randn(self.pred_len, self.d_model) * scale
        self.bias = np.zeros(self.pred_len)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Parameters:
        -----------
        x : np.ndarray
            输入特征，shape为 (batch, d_model) 或 (batch, seq_len, d_model)
            
        Returns:
        --------
        output : np.ndarray
            预测输出，shape为 (batch, pred_len) 或 (batch, seq_len, pred_len)
        """
        return x @ self.weight.T + self.bias
    
    def load_weights(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """加载权重"""
        self.weight = weight
        if bias is not None:
            self.bias = bias
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """获取权重"""
        return {'weight': self.weight, 'bias': self.bias}
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class ImputationHead:
    """
    缺失值填充头
    
    Imputation head for missing value completion.
    Structure: Linear(d_model, seq_len)
    
    Reference:
        self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
    """
    
    def __init__(self, d_model: int, seq_len: int):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        seq_len : int
            序列长度 / Sequence length
        """
        self.d_model = d_model
        self.seq_len = seq_len
        self.weight = None
        self.bias = None
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        scale = np.sqrt(2.0 / (self.d_model + self.seq_len))
        self.weight = np.random.randn(self.seq_len, self.d_model) * scale
        self.bias = np.zeros(self.seq_len)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Parameters:
        -----------
        x : np.ndarray
            输入特征，shape为 (batch, d_model) 或 (batch, n_features, d_model)
            
        Returns:
        --------
        output : np.ndarray
            填充输出，shape为 (batch, seq_len) 或 (batch, n_features, seq_len)
        """
        return x @ self.weight.T + self.bias
    
    def load_weights(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """加载权重"""
        self.weight = weight
        if bias is not None:
            self.bias = bias
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """获取权重"""
        return {'weight': self.weight, 'bias': self.bias}
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class AnomalyDetectionHead:
    """
    异常检测头
    
    Anomaly detection head for point-level anomaly detection.
    Structure: Linear(d_model, seq_len)
    
    Reference:
        self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
    """
    
    def __init__(self, d_model: int, seq_len: int):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        seq_len : int
            序列长度 / Sequence length
        """
        self.d_model = d_model
        self.seq_len = seq_len
        self.weight = None
        self.bias = None
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        scale = np.sqrt(2.0 / (self.d_model + self.seq_len))
        self.weight = np.random.randn(self.seq_len, self.d_model) * scale
        self.bias = np.zeros(self.seq_len)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Parameters:
        -----------
        x : np.ndarray
            输入特征，shape为 (batch, d_model) 或 (batch, n_features, d_model)
            
        Returns:
        --------
        output : np.ndarray
            异常得分，shape为 (batch, seq_len) 或 (batch, n_features, seq_len)
        """
        return x @ self.weight.T + self.bias
    
    def load_weights(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """加载权重"""
        self.weight = weight
        if bias is not None:
            self.bias = bias
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """获取权重"""
        return {'weight': self.weight, 'bias': self.bias}
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class ClassificationHead:
    """
    分类头
    
    Classification head for time series classification.
    Structure: GELU + Dropout + Linear(d_model * enc_in, num_class)
    
    Reference:
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
    """
    
    def __init__(
        self,
        d_model: int,
        enc_in: int,
        num_class: int,
        dropout_rate: float = 0.1
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        enc_in : int
            输入特征数 / Number of input features
        num_class : int
            分类类别数 / Number of classes
        dropout_rate : float, default=0.1
            Dropout 比率
        """
        self.d_model = d_model
        self.enc_in = enc_in
        self.num_class = num_class
        self.dropout_rate = dropout_rate
        self.input_dim = d_model * enc_in
        self.weight = None
        self.bias = None
        self.training = False
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        scale = np.sqrt(2.0 / (self.input_dim + self.num_class))
        self.weight = np.random.randn(self.num_class, self.input_dim) * scale
        self.bias = np.zeros(self.num_class)
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Parameters:
        -----------
        x : np.ndarray
            输入特征，支持以下形状:
            - (batch, d_model * enc_in): 已展平的特征
            - (batch, enc_in, d_model): 3D特征，将被展平为 (batch, enc_in * d_model)
            
        Returns:
        --------
        output : np.ndarray
            分类 logits，shape为 (batch, num_class)
            
        Raises:
        -------
        ValueError
            如果输入形状不符合预期
        """
        # Validate and flatten input
        if x.ndim == 3:
            batch_size = x.shape[0]
            expected_flat_dim = x.shape[1] * x.shape[2]
            if expected_flat_dim != self.input_dim:
                raise ValueError(
                    f"Input shape {x.shape} flattens to {expected_flat_dim}, "
                    f"but expected input_dim={self.input_dim} (d_model={self.d_model} * enc_in={self.enc_in})"
                )
            x = x.reshape(batch_size, -1)
        elif x.ndim == 2:
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"Input shape {x.shape} has dimension {x.shape[1]}, "
                    f"but expected input_dim={self.input_dim} (d_model={self.d_model} * enc_in={self.enc_in})"
                )
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")
        
        # GELU activation
        x = gelu(x)
        
        # Dropout
        x = dropout(x, self.dropout_rate, self.training)
        
        # Linear projection
        return x @ self.weight.T + self.bias
    
    def load_weights(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """加载权重"""
        self.weight = weight
        if bias is not None:
            self.bias = bias
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """获取权重"""
        return {'weight': self.weight, 'bias': self.bias}
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# Task name constants
TASK_LONG_TERM_FORECAST = 'long_term_forecast'
TASK_SHORT_TERM_FORECAST = 'short_term_forecast'
TASK_IMPUTATION = 'imputation'
TASK_ANOMALY_DETECTION = 'anomaly_detection'
TASK_CLASSIFICATION = 'classification'


def create_task_head(
    task_name: str,
    d_model: int,
    **kwargs
) -> object:
    """
    根据任务类型创建对应的预测头
    Create task-specific head based on task name
    
    Parameters:
    -----------
    task_name : str
        任务名称:
        - 'long_term_forecast' / 'short_term_forecast': 需要 pred_len
        - 'imputation': 需要 seq_len
        - 'anomaly_detection': 需要 seq_len
        - 'classification': 需要 enc_in, num_class
    d_model : int
        模型维度
    **kwargs : dict
        任务特定参数
        
    Returns:
    --------
    head : object
        任务特定的预测头
    """
    if task_name in [TASK_LONG_TERM_FORECAST, TASK_SHORT_TERM_FORECAST]:
        pred_len = kwargs.get('pred_len')
        if pred_len is None:
            raise ValueError(f"pred_len is required for {task_name}")
        return ForecastHead(d_model, pred_len)
    
    elif task_name == TASK_IMPUTATION:
        seq_len = kwargs.get('seq_len')
        if seq_len is None:
            raise ValueError(f"seq_len is required for {task_name}")
        return ImputationHead(d_model, seq_len)
    
    elif task_name == TASK_ANOMALY_DETECTION:
        seq_len = kwargs.get('seq_len')
        if seq_len is None:
            raise ValueError(f"seq_len is required for {task_name}")
        return AnomalyDetectionHead(d_model, seq_len)
    
    elif task_name == TASK_CLASSIFICATION:
        enc_in = kwargs.get('enc_in')
        num_class = kwargs.get('num_class')
        dropout_rate = kwargs.get('dropout', 0.1)
        if enc_in is None or num_class is None:
            raise ValueError(f"enc_in and num_class are required for {task_name}")
        return ClassificationHead(d_model, enc_in, num_class, dropout_rate)
    
    else:
        raise ValueError(f"Unknown task name: {task_name}")


__all__ = [
    'gelu',
    'dropout',
    'ForecastHead',
    'ImputationHead',
    'AnomalyDetectionHead',
    'ClassificationHead',
    'create_task_head',
    'TASK_LONG_TERM_FORECAST',
    'TASK_SHORT_TERM_FORECAST',
    'TASK_IMPUTATION',
    'TASK_ANOMALY_DETECTION',
    'TASK_CLASSIFICATION',
]
