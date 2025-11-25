"""
Feed Forward Network Layers

前馈网络层
"""

import numpy as np
from typing import Optional


class FeedForward:
    """
    前馈网络层
    
    Standard feed-forward network with two linear layers and activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        d_ff : int
            前馈网络隐藏层维度 / Feed-forward hidden dimension
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        activation : str, default='relu'
            激活函数 ('relu', 'gelu') / Activation function
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        # Weights
        self.linear1_weight = None
        self.linear1_bias = None
        self.linear2_weight = None
        self.linear2_bias = None
    
    def set_weights(
        self,
        linear1_weight: Optional[np.ndarray] = None,
        linear1_bias: Optional[np.ndarray] = None,
        linear2_weight: Optional[np.ndarray] = None,
        linear2_bias: Optional[np.ndarray] = None
    ):
        """设置权重"""
        self.linear1_weight = linear1_weight
        self.linear1_bias = linear1_bias
        self.linear2_weight = linear2_weight
        self.linear2_bias = linear2_bias
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """应用激活函数"""
        if self.activation == 'relu':
            return self._relu(x)
        elif self.activation == 'gelu':
            return self._gelu(x)
        else:
            return x
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch, seq_len, d_model)
            
        Returns:
        --------
        output : np.ndarray
            输出，shape为 (batch, seq_len, d_model)
        """
        # First linear layer
        if self.linear1_weight is not None:
            output = x @ self.linear1_weight.T
            if self.linear1_bias is not None:
                output = output + self.linear1_bias
        else:
            # Identity-like transformation
            output = x
        
        # Activation
        output = self._activate(output)
        
        # Second linear layer
        if self.linear2_weight is not None:
            output = output @ self.linear2_weight.T
            if self.linear2_bias is not None:
                output = output + self.linear2_bias
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)


class ConvFeedForward:
    """
    卷积前馈网络层
    
    Feed-forward network using 1D convolutions instead of linear layers.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        kernel_size: int = 1,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        d_ff : int
            前馈网络隐藏层维度 / Feed-forward hidden dimension
        kernel_size : int, default=1
            卷积核大小 / Kernel size
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        activation : str, default='relu'
            激活函数 / Activation function
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        
        # Weights
        self.conv1_weight = None
        self.conv1_bias = None
        self.conv2_weight = None
        self.conv2_bias = None
    
    def set_weights(
        self,
        conv1_weight: Optional[np.ndarray] = None,
        conv1_bias: Optional[np.ndarray] = None,
        conv2_weight: Optional[np.ndarray] = None,
        conv2_bias: Optional[np.ndarray] = None
    ):
        """设置权重"""
        self.conv1_weight = conv1_weight
        self.conv1_bias = conv1_bias
        self.conv2_weight = conv2_weight
        self.conv2_bias = conv2_bias
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """应用激活函数"""
        if self.activation == 'relu':
            return self._relu(x)
        elif self.activation == 'gelu':
            return self._gelu(x)
        else:
            return x
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        """
        # For kernel_size=1, equivalent to linear
        if self.conv1_weight is not None:
            output = x @ self.conv1_weight.T
            if self.conv1_bias is not None:
                output = output + self.conv1_bias
        else:
            output = x
        
        output = self._activate(output)
        
        if self.conv2_weight is not None:
            output = output @ self.conv2_weight.T
            if self.conv2_bias is not None:
                output = output + self.conv2_bias
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)


class GatedLinearUnit:
    """
    门控线性单元
    
    Gated Linear Unit (GLU) for more expressive feed-forward networks.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        d_ff : int
            前馈网络隐藏层维度 / Feed-forward hidden dimension
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Weights
        self.linear_weight = None
        self.linear_bias = None
        self.gate_weight = None
        self.gate_bias = None
        self.output_weight = None
        self.output_bias = None
    
    def set_weights(
        self,
        linear_weight: Optional[np.ndarray] = None,
        linear_bias: Optional[np.ndarray] = None,
        gate_weight: Optional[np.ndarray] = None,
        gate_bias: Optional[np.ndarray] = None,
        output_weight: Optional[np.ndarray] = None,
        output_bias: Optional[np.ndarray] = None
    ):
        """设置权重"""
        self.linear_weight = linear_weight
        self.linear_bias = linear_bias
        self.gate_weight = gate_weight
        self.gate_bias = gate_bias
        self.output_weight = output_weight
        self.output_bias = output_bias
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        """
        # Linear transformation
        if self.linear_weight is not None:
            linear_out = x @ self.linear_weight.T
            if self.linear_bias is not None:
                linear_out = linear_out + self.linear_bias
        else:
            linear_out = x
        
        # Gate
        if self.gate_weight is not None:
            gate = x @ self.gate_weight.T
            if self.gate_bias is not None:
                gate = gate + self.gate_bias
            gate = self._sigmoid(gate)
        else:
            gate = np.ones_like(linear_out) * 0.5
        
        # Gated output
        output = linear_out * gate
        
        # Output projection
        if self.output_weight is not None:
            output = output @ self.output_weight.T
            if self.output_bias is not None:
                output = output + self.output_bias
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)
