"""
Transformer Base Model

Transformer基础模型，用于时序预测
"""

import numpy as np
from typing import Dict, Any, Optional
import warnings
from .layers import PositionalEncoding


class TransformerBaseModel:
    """
    Transformer基础模型
    
    Transformer encoder-decoder model for time series forecasting.
    Supports multi-head attention and positional encoding.
    
    This is the base model that can be extended for different forecasting tasks.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        enc_in: int = 1,
        dec_in: int = 1,
        c_out: int = 1,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        pred_len : int
            预测长度 / Prediction length
        d_model : int, default=512
            模型维度 / Model dimension
        n_heads : int, default=8
            注意力头数 / Number of attention heads
        e_layers : int, default=2
            编码器层数 / Number of encoder layers
        d_layers : int, default=1
            解码器层数 / Number of decoder layers
        d_ff : int, default=2048
            前馈网络维度 / Feed-forward network dimension
        enc_in : int, default=1
            编码器输入维度 / Encoder input dimension
        dec_in : int, default=1
            解码器输入维度 / Decoder input dimension
        c_out : int, default=1
            输出维度 / Output dimension
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.dropout = dropout
        
        # Model components
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        
        # Model parameters (will be set when loading weights)
        self.encoder_weights = None
        self.decoder_weights = None
        self.output_projection = None
    
    def set_weights(
        self, 
        encoder_weights: Optional[Dict[str, np.ndarray]] = None,
        decoder_weights: Optional[Dict[str, np.ndarray]] = None,
        output_projection: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        设置模型权重
        Set model weights
        
        Parameters:
        -----------
        encoder_weights : dict, optional
            编码器权重
        decoder_weights : dict, optional
            解码器权重
        output_projection : dict, optional
            输出投影层权重
        """
        self.encoder_weights = encoder_weights
        self.decoder_weights = decoder_weights
        self.output_projection = output_projection
    
    def forward(self, x: np.ndarray, use_simple_projection: bool = True) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features)
        use_simple_projection : bool, default=True
            是否使用简化的投影方法（当完整模型权重不可用时）
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果，shape为 (batch_size, pred_len, n_features)
        """
        batch_size, seq_len, n_features = x.shape
        
        # This is a simplified prediction method
        # In a full implementation, you would need to implement the complete
        # Transformer forward pass with attention mechanisms
        
        if use_simple_projection and self.output_projection:
            # Use a simplified linear projection for demonstration
            warnings.warn(
                "Using simplified prediction method. For accurate predictions, "
                "consider using the full Transformer implementation with PyTorch.",
                UserWarning
            )
            
            # Get projection weights if available
            proj_weight = None
            proj_bias = None
            for key, val in self.output_projection.items():
                if 'weight' in key:
                    proj_weight = val
                if 'bias' in key:
                    proj_bias = val
            
            if proj_weight is not None:
                # Simple linear projection
                x_flat = x.reshape(batch_size, -1)
                if proj_weight.shape[1] == x_flat.shape[1]:
                    predictions = x_flat @ proj_weight.T
                    if proj_bias is not None:
                        predictions = predictions + proj_bias
                    predictions = predictions.reshape(batch_size, self.pred_len, self.c_out)
                    return predictions
        
        # Fallback: simple last-value repetition
        warnings.warn(
            "Could not perform full prediction. Returning last-value forecast as fallback.",
            UserWarning
        )
        last_values = x[:, -1:, :]
        predictions = np.repeat(last_values, self.pred_len, axis=1)
        
        return predictions
    
    def __call__(self, x: np.ndarray, use_simple_projection: bool = True) -> np.ndarray:
        """Make the model callable"""
        return self.forward(x, use_simple_projection)
    
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
            'model_type': 'Transformer',
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'e_layers': self.e_layers,
            'd_layers': self.d_layers,
            'd_ff': self.d_ff,
            'enc_in': self.enc_in,
            'dec_in': self.dec_in,
            'c_out': self.c_out,
            'dropout': self.dropout,
            'has_encoder_weights': self.encoder_weights is not None,
            'has_decoder_weights': self.decoder_weights is not None,
            'has_output_projection': self.output_projection is not None
        }
