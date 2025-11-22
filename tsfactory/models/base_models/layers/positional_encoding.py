"""
Positional Encoding Layer

位置编码层，用于Transformer模型
"""

import numpy as np


class PositionalEncoding:
    """
    位置编码层
    
    Positional encoding layer for Transformer models.
    Generates sinusoidal position encodings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        max_len : int, default=5000
            最大序列长度 / Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len
        
        # Pre-compute positional encodings
        self.pe = self._generate_positional_encoding(max_len, d_model)
    
    def _generate_positional_encoding(self, length: int, d_model: int) -> np.ndarray:
        """
        生成位置编码
        Generate positional encoding
        
        Parameters:
        -----------
        length : int
            序列长度
        d_model : int
            模型维度
            
        Returns:
        --------
        pe : np.ndarray
            位置编码，shape为 (length, d_model)
        """
        pe = np.zeros((length, d_model))
        position = np.arange(0, length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Assign sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Assign cosine to odd indices
        # When d_model is odd, there are fewer odd indices than div_term elements
        n_cos_positions = (d_model - 1) // 2 + (d_model % 2 == 0)
        pe[:, 1::2] = np.cos(position * div_term[:n_cos_positions])
        
        return pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播：添加位置编码
        Forward pass: add positional encoding to input
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, d_model)
            
        Returns:
        --------
        x_with_pe : np.ndarray
            添加位置编码后的序列
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_len:
            # Generate new PE if sequence is longer than max_len
            pe = self._generate_positional_encoding(seq_len, d_model)
        else:
            pe = self.pe[:seq_len, :]
        
        # Add positional encoding (broadcast over batch dimension)
        return x + pe[np.newaxis, :, :]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)
    
    def get_encoding(self, length: int) -> np.ndarray:
        """
        获取位置编码
        Get positional encoding for a specific length
        
        Parameters:
        -----------
        length : int
            序列长度
            
        Returns:
        --------
        pe : np.ndarray
            位置编码，shape为 (length, d_model)
        """
        if length > self.max_len:
            return self._generate_positional_encoding(length, self.d_model)
        return self.pe[:length, :]
