"""
Deep Learning Forecasting Models

深度学习预测模型，继承自base_models并添加预测任务的功能
"""

import numpy as np
from typing import Dict, Any
from ..base_models import DLinearBaseModel, TransformerBaseModel
from ...core.dl_data_loader import DLinearDataLoader, TransformerDataLoader


class DLinearForecaster:
    """
    DLinear预测模型
    
    Forecasting model using DLinear base model with additional task-specific functionality.
    Inherits from DLinearBaseModel and adds forecasting-specific heads and utilities.
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
            移动平均核大小 / Moving average kernel size for decomposition
        """
        # Initialize base model
        self.base_model = DLinearBaseModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            individual=individual,
            kernel_size=kernel_size
        )
        self.is_loaded = False
    
    # Properties for backward compatibility and testing
    @property
    def seq_len(self):
        return self.base_model.seq_len
    
    @property
    def pred_len(self):
        return self.base_model.pred_len
    
    @property
    def enc_in(self):
        return self.base_model.enc_in
    
    @property
    def individual(self):
        return self.base_model.individual
    
    @property
    def kernel_size(self):
        return self.base_model.kernel_size
    
    @property
    def seasonal_weights(self):
        return self.base_model.seasonal_linear
    
    @property
    def trend_weights(self):
        return self.base_model.trend_linear
    
    def _moving_average(self, x: np.ndarray, kernel_size: int) -> np.ndarray:
        """Delegate to decomposition layer"""
        decomp = self.base_model.decomposition
        old_kernel = decomp.kernel_size
        decomp.kernel_size = kernel_size
        result = decomp._moving_average(x)
        decomp.kernel_size = old_kernel
        return result
    
    def _series_decomp(self, x: np.ndarray):
        """Delegate to decomposition layer"""
        return self.base_model.decomposition.forward(x)
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'DLinearForecaster':
        """
        从字典加载模型参数
        Load model parameters from dictionary
        
        Parameters:
        -----------
        state_dict : dict
            包含模型权重的字典
            
        Returns:
        --------
        self : DLinearForecaster
        """
        try:
            weights = DLinearDataLoader.load_from_dict(state_dict)
            
            seasonal_weights = weights.get('seasonal')
            trend_weights = weights.get('trend')
            
            self.base_model.set_weights(seasonal_weights, trend_weights)
            self.is_loaded = True
            return self
            
        except Exception as e:
            raise ValueError(f"Failed to load DLinear weights: {str(e)}")
    
    def load_from_file(self, filepath: str) -> 'DLinearForecaster':
        """
        从文件加载模型
        Load model from file
        
        Parameters:
        -----------
        filepath : str
            模型文件路径 / Path to model file (.npy, .npz, or PyTorch .pth/.pt)
            
        Returns:
        --------
        self : DLinearForecaster
        """
        try:
            weights = DLinearDataLoader.load_from_file(filepath)
            
            seasonal_weights = weights.get('seasonal')
            trend_weights = weights.get('trend')
            
            self.base_model.set_weights(seasonal_weights, trend_weights)
            self.is_loaded = True
            return self
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {str(e)}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        进行预测
        Make predictions
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features) 或 (seq_len, n_features) 或 (seq_len,)
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction. Call load_from_dict() or load_from_file() first.")
        
        # Normalize input shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, n_features = x.shape
        
        if seq_len != self.base_model.seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) does not match expected length ({self.base_model.seq_len})")
        
        # Forward pass through base model
        predictions = self.base_model.forward(x)
        
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
        info = self.base_model.get_model_info()
        info['is_loaded'] = self.is_loaded
        info['task'] = 'forecasting'
        return info


class TransformerForecaster:
    """
    Transformer预测模型
    
    Forecasting model using Transformer base model with additional task-specific functionality.
    Inherits from TransformerBaseModel and adds forecasting-specific heads and utilities.
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
        # Initialize base model
        self.base_model = TransformerBaseModel(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            dropout=dropout
        )
        self.is_loaded = False
    
    # Properties for backward compatibility and testing
    @property
    def seq_len(self):
        return self.base_model.seq_len
    
    @property
    def pred_len(self):
        return self.base_model.pred_len
    
    @property
    def d_model(self):
        return self.base_model.d_model
    
    @property
    def n_heads(self):
        return self.base_model.n_heads
    
    @property
    def e_layers(self):
        return self.base_model.e_layers
    
    @property
    def d_layers(self):
        return self.base_model.d_layers
    
    @property
    def d_ff(self):
        return self.base_model.d_ff
    
    @property
    def enc_in(self):
        return self.base_model.enc_in
    
    @property
    def dec_in(self):
        return self.base_model.dec_in
    
    @property
    def c_out(self):
        return self.base_model.c_out
    
    @property
    def dropout(self):
        return self.base_model.dropout
    
    @property
    def encoder_weights(self):
        return self.base_model.encoder_weights
    
    @property
    def decoder_weights(self):
        return self.base_model.decoder_weights
    
    @property
    def output_projection(self):
        return self.base_model.output_projection
    
    def _positional_encoding(self, length: int, d_model: int) -> np.ndarray:
        """Delegate to positional encoding layer"""
        pe_layer = self.base_model.positional_encoding
        return pe_layer._generate_positional_encoding(length, d_model)
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'TransformerForecaster':
        """
        从字典加载模型参数
        Load model parameters from dictionary
        
        Parameters:
        -----------
        state_dict : dict
            包含模型权重的字典
            
        Returns:
        --------
        self : TransformerForecaster
        """
        try:
            weights = TransformerDataLoader.load_from_dict(state_dict)
            
            encoder_weights = weights.get('encoder')
            decoder_weights = weights.get('decoder')
            projection_weights = weights.get('projection')
            
            self.base_model.set_weights(encoder_weights, decoder_weights, projection_weights)
            self.is_loaded = True
            return self
            
        except Exception as e:
            raise ValueError(f"Failed to load Transformer weights: {str(e)}")
    
    def load_from_file(self, filepath: str) -> 'TransformerForecaster':
        """
        从文件加载模型
        Load model from file
        
        Parameters:
        -----------
        filepath : str
            模型文件路径 / Path to model file (.npy, .npz, or PyTorch .pth/.pt)
            
        Returns:
        --------
        self : TransformerForecaster
        """
        try:
            weights = TransformerDataLoader.load_from_file(filepath)
            
            encoder_weights = weights.get('encoder')
            decoder_weights = weights.get('decoder')
            projection_weights = weights.get('projection')
            
            self.base_model.set_weights(encoder_weights, decoder_weights, projection_weights)
            self.is_loaded = True
            return self
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {str(e)}")
    
    def predict(self, x: np.ndarray, use_simple_projection: bool = True) -> np.ndarray:
        """
        进行预测
        Make predictions
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch_size, seq_len, n_features) 或 (seq_len, n_features) 或 (seq_len,)
        use_simple_projection : bool, default=True
            是否使用简化的投影方法
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction. Call load_from_dict() or load_from_file() first.")
        
        # Normalize input shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, n_features = x.shape
        
        if seq_len != self.base_model.seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) does not match expected length ({self.base_model.seq_len})")
        
        # Forward pass through base model
        predictions = self.base_model.forward(x, use_simple_projection)
        
        # Squeeze dimensions appropriately
        if batch_size == 1 and self.base_model.c_out == 1:
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
        info = self.base_model.get_model_info()
        info['is_loaded'] = self.is_loaded
        info['task'] = 'forecasting'
        return info
