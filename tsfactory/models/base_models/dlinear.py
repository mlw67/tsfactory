"""
DLinear Base Model

DLinear基础模型，用于时序预测
"""

import numpy as np
from typing import Dict, Any, Tuple
from .layers import SeriesDecomp


class DLinearBaseModel:
    """
    DLinear基础模型
    
    DLinear is a simple yet effective time series forecasting model that decomposes 
    the time series into trend and seasonal components, then uses linear layers 
    for prediction separately.
    
    This is the base model that can be extended for different forecasting tasks.
    """
    
    def __init__(
        self, 
        seq_len: int,
        pred_len: int,
        enc_in: int = 1,
        individual: bool = False,
        kernel_size: int = 25
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        pred_len : int
            预测长度 / Prediction length
        enc_in : int, default=1
            输入特征维度 / Number of input features (channels)
        individual : bool, default=False
            是否为每个特征使用独立的线性层 / Whether to use individual linear layers for each channel
        kernel_size : int, default=25
            移动平均核大小，用于趋势分解 / Moving average kernel size for decomposition
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.kernel_size = kernel_size
        
        # Model components
        self.decomposition = SeriesDecomp(kernel_size=kernel_size)
        
        # Model parameters (will be set when loading weights)
        self.seasonal_linear = None
        self.trend_linear = None
    
    def set_weights(self, seasonal_weights: Dict[str, np.ndarray], trend_weights: Dict[str, np.ndarray]):
        """
        设置模型权重
        Set model weights
        
        Parameters:
        -----------
        seasonal_weights : dict
            季节性线性层权重，包含 'weight' 和 'bias'
        trend_weights : dict
            趋势线性层权重，包含 'weight' 和 'bias'
        """
        self.seasonal_linear = seasonal_weights
        self.trend_linear = trend_weights
    
    def _linear_transform(self, x: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """
        线性变换
        Apply linear transformation
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch_size, seq_len * n_features)
        weights : dict
            权重字典，包含 'weight' 和 'bias'
            
        Returns:
        --------
        output : np.ndarray
            输出，shape为 (batch_size, pred_len * n_features)
        """
        output = x @ weights['weight'].T
        if 'bias' in weights:
            output = output + weights['bias']
        return output
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features)
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果，shape为 (batch_size, pred_len, n_features)
        """
        if self.seasonal_linear is None and self.trend_linear is None:
            raise ValueError("Model weights not set. Call set_weights() first.")
        
        batch_size, seq_len, n_features = x.shape
        
        # Decompose the series
        seasonal, trend = self.decomposition(x)
        
        # Predict seasonal component
        if self.seasonal_linear is not None:
            seasonal_flat = seasonal.reshape(batch_size, -1)
            seasonal_pred = self._linear_transform(seasonal_flat, self.seasonal_linear)
            seasonal_pred = seasonal_pred.reshape(batch_size, self.pred_len, n_features)
        else:
            seasonal_pred = np.zeros((batch_size, self.pred_len, n_features))
        
        # Predict trend component
        if self.trend_linear is not None:
            trend_flat = trend.reshape(batch_size, -1)
            trend_pred = self._linear_transform(trend_flat, self.trend_linear)
            trend_pred = trend_pred.reshape(batch_size, self.pred_len, n_features)
        else:
            trend_pred = np.zeros((batch_size, self.pred_len, n_features))
        
        # Combine predictions
        predictions = seasonal_pred + trend_pred
        
        return predictions
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the model callable"""
        return self.forward(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        Get model information
        
        Returns:
        --------
        info : dict
            模型配置信息
        """
        return {
            'model_type': 'DLinear',
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.enc_in,
            'individual': self.individual,
            'kernel_size': self.kernel_size,
            'has_seasonal_weights': self.seasonal_linear is not None,
            'has_trend_weights': self.trend_linear is not None
        }
