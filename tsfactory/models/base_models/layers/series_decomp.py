"""
Series Decomposition Layer

时序分解层，用于将时间序列分解为趋势和季节性成分
"""

import numpy as np
from typing import Tuple


class SeriesDecomp:
    """
    时序分解层
    
    Series decomposition layer that separates time series into trend and seasonal components.
    Uses moving average to extract trend component.
    """
    
    def __init__(self, kernel_size: int = 25):
        """
        Parameters:
        -----------
        kernel_size : int, default=25
            移动平均核大小 / Moving average kernel size for trend extraction
        """
        self.kernel_size = kernel_size
    
    def _moving_average(self, x: np.ndarray) -> np.ndarray:
        """
        计算移动平均以提取趋势
        Compute moving average to extract trend
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features)
            
        Returns:
        --------
        trend : np.ndarray
            趋势分量
        """
        batch_size, seq_len, n_features = x.shape
        
        # Pad the sequence
        pad_size = self.kernel_size // 2
        x_padded = np.pad(x, ((0, 0), (pad_size, pad_size), (0, 0)), mode='edge')
        
        # Compute moving average for each feature
        trend = np.zeros_like(x)
        for i in range(n_features):
            for j in range(seq_len):
                trend[:, j, i] = np.mean(x_padded[:, j:j+self.kernel_size, i], axis=1)
        
        return trend
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播：时序分解
        Forward pass: decompose series into seasonal and trend components
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features)
            
        Returns:
        --------
        seasonal : np.ndarray
            季节性分量
        trend : np.ndarray
            趋势分量
        """
        trend = self._moving_average(x)
        seasonal = x - trend
        return seasonal, trend
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make the layer callable"""
        return self.forward(x)
