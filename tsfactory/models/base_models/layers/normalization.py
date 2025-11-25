"""
Normalization Layers

归一化层
"""

import numpy as np
from typing import Optional, Tuple


class LayerNorm:
    """
    层归一化
    
    Layer normalization for stabilizing training.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        eps : float, default=1e-6
            数值稳定性参数 / Epsilon for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def set_weights(self, gamma: np.ndarray, beta: np.ndarray):
        """设置权重"""
        self.gamma = gamma
        self.beta = beta
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (..., d_model)
            
        Returns:
        --------
        output : np.ndarray
            归一化输出
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        
        output = (x - mean) / (std + self.eps)
        output = output * self.gamma + self.beta
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)


class RevIN:
    """
    可逆实例归一化
    
    Reversible Instance Normalization for handling distribution shift.
    Commonly used in time series models.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        """
        Parameters:
        -----------
        num_features : int
            特征数量 / Number of features
        eps : float, default=1e-5
            数值稳定性参数 / Epsilon for numerical stability
        affine : bool, default=True
            是否使用仿射变换 / Whether to use affine transformation
        """
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        # Statistics (computed during normalization)
        self.mean = None
        self.std = None
        
        # Learnable parameters
        if affine:
            self.affine_weight = np.ones(num_features)
            self.affine_bias = np.zeros(num_features)
        else:
            self.affine_weight = None
            self.affine_bias = None
    
    def set_weights(self, weight: np.ndarray, bias: np.ndarray):
        """设置仿射变换权重"""
        self.affine_weight = weight
        self.affine_bias = bias
    
    def normalize(self, x: np.ndarray, mode: str = 'norm') -> np.ndarray:
        """
        归一化
        Normalize
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch, seq_len, n_features)
        mode : str, default='norm'
            模式：'norm' 归一化，'denorm' 反归一化
            
        Returns:
        --------
        output : np.ndarray
            输出
        """
        if mode == 'norm':
            # Compute and store statistics
            self.mean = np.mean(x, axis=1, keepdims=True)
            self.std = np.std(x, axis=1, keepdims=True) + self.eps
            
            # Normalize
            output = (x - self.mean) / self.std
            
            # Apply affine transformation
            if self.affine:
                output = output * self.affine_weight + self.affine_bias
            
            return output
        
        elif mode == 'denorm':
            if self.mean is None or self.std is None:
                raise ValueError("Must call normalize with mode='norm' first")
            
            # Remove affine transformation
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            
            # Denormalize
            output = x * self.std + self.mean
            
            return output
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播（归一化）"""
        return self.normalize(x, mode='norm')
    
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """反向传播（反归一化）"""
        return self.normalize(x, mode='denorm')
    
    def __call__(self, x: np.ndarray, mode: str = 'norm') -> np.ndarray:
        """Make the layer callable"""
        return self.normalize(x, mode)


class BatchNorm1d:
    """
    一维批归一化
    
    Batch normalization for 1D sequences.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        """
        Parameters:
        -----------
        num_features : int
            特征数量 / Number of features
        eps : float, default=1e-5
            数值稳定性参数 / Epsilon for numerical stability
        momentum : float, default=0.1
            动量 / Momentum for running statistics
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def set_weights(self, gamma: np.ndarray, beta: np.ndarray):
        """设置权重"""
        self.gamma = gamma
        self.beta = beta
    
    def set_running_stats(self, mean: np.ndarray, var: np.ndarray):
        """设置运行统计"""
        self.running_mean = mean
        self.running_var = var
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch, n_features) 或 (batch, seq_len, n_features)
        training : bool, default=False
            是否为训练模式
            
        Returns:
        --------
        output : np.ndarray
            归一化输出
        """
        if training:
            # Use batch statistics
            if x.ndim == 2:
                mean = np.mean(x, axis=0)
                var = np.var(x, axis=0)
            else:
                mean = np.mean(x, axis=(0, 1))
                var = np.var(x, axis=(0, 1))
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        output = (x - mean) / np.sqrt(var + self.eps)
        output = output * self.gamma + self.beta
        
        return output
    
    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x, training)


class RMSNorm:
    """
    RMS归一化
    
    Root Mean Square Layer Normalization, simpler alternative to LayerNorm.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        eps : float, default=1e-6
            数值稳定性参数 / Epsilon for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Scale parameter
        self.scale = np.ones(d_model)
    
    def set_weights(self, scale: np.ndarray):
        """设置权重"""
        self.scale = scale
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (..., d_model)
            
        Returns:
        --------
        output : np.ndarray
            归一化输出
        """
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        output = x / rms * self.scale
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)
