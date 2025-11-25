"""
NLinear Base Model

NLinear基础模型，用于时序预测
"""

import numpy as np
from typing import Dict, Any, Optional


class NLinearBaseModel:
    """
    NLinear基础模型
    
    NLinear is a simple yet effective time series forecasting model that normalizes 
    the input by subtracting the last value, then applies a linear layer for prediction,
    and finally adds the last value back.
    
    This normalization scheme helps the model handle distribution shift.
    """
    
    def __init__(
        self, 
        seq_len: int,
        pred_len: int,
        enc_in: int = 1,
        individual: bool = False
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
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        
        # Model parameters (will be set when loading weights)
        self.linear = None
    
    def set_weights(self, linear_weights: Dict[str, np.ndarray]):
        """
        设置模型权重
        Set model weights
        
        Parameters:
        -----------
        linear_weights : dict
            线性层权重，包含 'weight' 和可选的 'bias'
        """
        self.linear = linear_weights
    
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
        if self.linear is None:
            raise ValueError("Model weights not set. Call set_weights() first.")
        
        batch_size, seq_len, n_features = x.shape
        
        # Get the last value for normalization
        last_value = x[:, -1:, :]  # (batch_size, 1, n_features)
        
        # Normalize by subtracting last value
        x_normalized = x - last_value
        
        # Flatten and apply linear
        x_flat = x_normalized.reshape(batch_size, -1)
        output = self._linear_transform(x_flat, self.linear)
        
        # Reshape
        output = output.reshape(batch_size, self.pred_len, n_features)
        
        # Add last value back
        predictions = output + last_value
        
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
            'model_type': 'NLinear',
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.enc_in,
            'individual': self.individual,
            'has_linear_weights': self.linear is not None
        }
