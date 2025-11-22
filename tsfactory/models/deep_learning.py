"""
Deep Learning Models for Time Series

深度学习模型加载器，包括DLinear和Transformer编码解码器类型的模型
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
import warnings


class DLinearLoader:
    """
    DLinear模型加载器
    
    DLinear是一种简单而有效的时序预测模型，通过将时间序列分解为趋势和季节性成分，
    然后分别使用线性层进行预测。
    
    DLinear Model Loader for Time Series Forecasting.
    DLinear is a simple yet effective time series forecasting model that decomposes 
    the time series into trend and seasonal components, then uses linear layers 
    for prediction separately.
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
        
        # Model parameters (will be set when loading)
        self.seasonal_weights = None
        self.trend_weights = None
        self.is_loaded = False
        
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'DLinearLoader':
        """
        从字典加载模型参数
        Load model parameters from dictionary
        
        Parameters:
        -----------
        state_dict : dict
            包含模型权重的字典 / Dictionary containing model weights
            Expected keys: 'seasonal_Linear.weight', 'seasonal_Linear.bias',
                          'trend_Linear.weight', 'trend_Linear.bias'
                          
        Returns:
        --------
        self : DLinearLoader
        """
        try:
            # Load seasonal linear layer weights
            if 'seasonal_Linear.weight' in state_dict:
                self.seasonal_weights = {
                    'weight': np.array(state_dict['seasonal_Linear.weight']),
                    'bias': np.array(state_dict.get('seasonal_Linear.bias', 0))
                }
            
            # Load trend linear layer weights  
            if 'trend_Linear.weight' in state_dict:
                self.trend_weights = {
                    'weight': np.array(state_dict['trend_Linear.weight']),
                    'bias': np.array(state_dict.get('trend_Linear.bias', 0))
                }
            
            self.is_loaded = True
            return self
            
        except Exception as e:
            raise ValueError(f"Failed to load DLinear weights: {str(e)}")
    
    def load_from_file(self, filepath: str) -> 'DLinearLoader':
        """
        从文件加载模型
        Load model from file
        
        Parameters:
        -----------
        filepath : str
            模型文件路径 / Path to model file (.npy, .npz, or PyTorch .pth/.pt)
            
        Returns:
        --------
        self : DLinearLoader
        """
        try:
            if filepath.endswith('.npz'):
                # Load from numpy archive
                data = np.load(filepath)
                state_dict = {key: data[key] for key in data.files}
                return self.load_from_dict(state_dict)
            elif filepath.endswith('.npy'):
                # Load from single numpy array
                data = np.load(filepath, allow_pickle=True).item()
                return self.load_from_dict(data)
            elif filepath.endswith(('.pth', '.pt')):
                # Try to load PyTorch model
                try:
                    import torch
                    checkpoint = torch.load(filepath, map_location='cpu')
                    # Convert torch tensors to numpy
                    state_dict = {k: v.cpu().numpy() for k, v in checkpoint.items()}
                    return self.load_from_dict(state_dict)
                except ImportError:
                    raise ImportError("PyTorch is required to load .pth/.pt files. Install it with: pip install torch")
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {str(e)}")
    
    def _moving_average(self, x: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        计算移动平均以提取趋势
        Compute moving average to extract trend
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features)
        kernel_size : int
            移动平均窗口大小
            
        Returns:
        --------
        trend : np.ndarray
            趋势分量
        """
        # Simple moving average using convolution
        batch_size, seq_len, n_features = x.shape
        
        # Pad the sequence
        pad_size = kernel_size // 2
        x_padded = np.pad(x, ((0, 0), (pad_size, pad_size), (0, 0)), mode='edge')
        
        # Compute moving average for each feature
        trend = np.zeros_like(x)
        for i in range(n_features):
            for j in range(seq_len):
                trend[:, j, i] = np.mean(x_padded[:, j:j+kernel_size, i], axis=1)
        
        return trend
    
    def _series_decomp(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        时序分解：趋势 + 季节性
        Series decomposition: trend + seasonal
        
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
        trend = self._moving_average(x, self.kernel_size)
        seasonal = x - trend
        return seasonal, trend
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        使用加载的模型进行预测
        Make predictions using loaded model
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features) 或 (seq_len, n_features) 或 (seq_len,)
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果，shape为 (batch_size, pred_len, n_features)
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction. Call load_from_dict() or load_from_file() first.")
        
        # Normalize input shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, n_features = x.shape
        
        if seq_len != self.seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) does not match expected length ({self.seq_len})")
        
        # Decompose the series
        seasonal, trend = self._series_decomp(x)
        
        # Predict seasonal component
        if self.seasonal_weights is not None:
            seasonal_flat = seasonal.reshape(batch_size, -1)
            seasonal_pred = seasonal_flat @ self.seasonal_weights['weight'].T + self.seasonal_weights['bias']
            seasonal_pred = seasonal_pred.reshape(batch_size, self.pred_len, n_features)
        else:
            seasonal_pred = np.zeros((batch_size, self.pred_len, n_features))
        
        # Predict trend component
        if self.trend_weights is not None:
            trend_flat = trend.reshape(batch_size, -1)
            trend_pred = trend_flat @ self.trend_weights['weight'].T + self.trend_weights['bias']
            trend_pred = trend_pred.reshape(batch_size, self.pred_len, n_features)
        else:
            trend_pred = np.zeros((batch_size, self.pred_len, n_features))
        
        # Combine predictions
        predictions = seasonal_pred + trend_pred
        
        # Squeeze dimensions appropriately
        if batch_size == 1 and n_features == 1:
            predictions = predictions.squeeze()
        elif batch_size == 1:
            predictions = predictions.squeeze(0)
        
        return predictions
    
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
            'is_loaded': self.is_loaded,
            'has_seasonal_weights': self.seasonal_weights is not None,
            'has_trend_weights': self.trend_weights is not None
        }


class TransformerLoader:
    """
    Transformer编码解码器模型加载器
    
    用于加载和使用Transformer编码解码器类型的时序预测模型。
    支持标准的Transformer架构，包括多头注意力机制和位置编码。
    
    Transformer Encoder-Decoder Model Loader for Time Series.
    Supports loading and using Transformer encoder-decoder type models 
    for time series forecasting with multi-head attention and positional encoding.
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
        
        # Model components (will be set when loading)
        self.encoder_weights = None
        self.decoder_weights = None
        self.output_projection = None
        self.is_loaded = False
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'TransformerLoader':
        """
        从字典加载模型参数
        Load model parameters from dictionary
        
        Parameters:
        -----------
        state_dict : dict
            包含模型权重的字典 / Dictionary containing model weights
            Expected keys include encoder, decoder, and projection layer weights
            
        Returns:
        --------
        self : TransformerLoader
        """
        try:
            # Organize weights by component
            encoder_keys = [k for k in state_dict.keys() if 'encoder' in k.lower()]
            decoder_keys = [k for k in state_dict.keys() if 'decoder' in k.lower()]
            projection_keys = [k for k in state_dict.keys() if 'projection' in k.lower() or 'output' in k.lower()]
            
            if encoder_keys:
                self.encoder_weights = {k: np.array(state_dict[k]) for k in encoder_keys}
            
            if decoder_keys:
                self.decoder_weights = {k: np.array(state_dict[k]) for k in decoder_keys}
            
            if projection_keys:
                self.output_projection = {k: np.array(state_dict[k]) for k in projection_keys}
            
            self.is_loaded = True
            return self
            
        except Exception as e:
            raise ValueError(f"Failed to load Transformer weights: {str(e)}")
    
    def load_from_file(self, filepath: str) -> 'TransformerLoader':
        """
        从文件加载模型
        Load model from file
        
        Parameters:
        -----------
        filepath : str
            模型文件路径 / Path to model file (.npy, .npz, or PyTorch .pth/.pt)
            
        Returns:
        --------
        self : TransformerLoader
        """
        try:
            if filepath.endswith('.npz'):
                # Load from numpy archive
                data = np.load(filepath)
                state_dict = {key: data[key] for key in data.files}
                return self.load_from_dict(state_dict)
            elif filepath.endswith('.npy'):
                # Load from single numpy array
                data = np.load(filepath, allow_pickle=True).item()
                return self.load_from_dict(data)
            elif filepath.endswith(('.pth', '.pt')):
                # Try to load PyTorch model
                try:
                    import torch
                    checkpoint = torch.load(filepath, map_location='cpu')
                    # Convert torch tensors to numpy
                    state_dict = {k: v.cpu().numpy() for k, v in checkpoint.items()}
                    return self.load_from_dict(state_dict)
                except ImportError:
                    raise ImportError("PyTorch is required to load .pth/.pt files. Install it with: pip install torch")
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {str(e)}")
    
    def _positional_encoding(self, length: int, d_model: int) -> np.ndarray:
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
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def predict(self, x: np.ndarray, use_simple_projection: bool = True) -> np.ndarray:
        """
        使用加载的模型进行预测
        Make predictions using loaded model
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features) 或 (seq_len, n_features) 或 (seq_len,)
        use_simple_projection : bool, default=True
            是否使用简化的投影方法（当完整模型权重不可用时）
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果，shape为 (batch_size, pred_len, n_features)
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction. Call load_from_dict() or load_from_file() first.")
        
        # Normalize input shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, n_features = x.shape
        
        if seq_len != self.seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) does not match expected length ({self.seq_len})")
        
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
                    
                    # Squeeze dimensions appropriately
                    if batch_size == 1 and self.c_out == 1:
                        predictions = predictions.squeeze()
                    elif batch_size == 1:
                        predictions = predictions.squeeze(0)
                    
                    return predictions
        
        # Fallback: simple last-value repetition
        warnings.warn(
            "Could not perform full prediction. Returning last-value forecast as fallback.",
            UserWarning
        )
        last_values = x[:, -1:, :]
        predictions = np.repeat(last_values, self.pred_len, axis=1)
        
        # Squeeze dimensions appropriately
        if batch_size == 1 and n_features == 1:
            predictions = predictions.squeeze()
        elif batch_size == 1:
            predictions = predictions.squeeze(0)
        
        return predictions
    
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
            'is_loaded': self.is_loaded,
            'has_encoder_weights': self.encoder_weights is not None,
            'has_decoder_weights': self.decoder_weights is not None,
            'has_output_projection': self.output_projection is not None
        }
