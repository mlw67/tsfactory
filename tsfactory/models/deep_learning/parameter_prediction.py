"""
Deep Learning Parameter Prediction Models

深度学习参数预测模型，用于短期预测
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from ..base_models import (
    DLinearBaseModel,
    NLinearBaseModel,
    TransformerBaseModel,
    AutoformerBaseModel,
    TimeXerBaseModel,
    TimeMixerBaseModel
)
from ...core.dl_data_loader import DLinearDataLoader, TransformerDataLoader


class NLinearForecaster:
    """
    NLinear预测模型
    
    Forecasting model using NLinear base model with normalization scheme.
    Effective for handling distribution shift in time series data.
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
            输入特征维度 / Number of input features
        individual : bool, default=False
            是否为每个特征使用独立的线性层
        """
        self.base_model = NLinearBaseModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            individual=individual
        )
        self.is_loaded = False
    
    @property
    def seq_len(self):
        return self.base_model.seq_len
    
    @property
    def pred_len(self):
        return self.base_model.pred_len
    
    @property
    def enc_in(self):
        return self.base_model.enc_in
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'NLinearForecaster':
        """从字典加载模型参数"""
        # Extract linear weights
        linear_weights = {}
        for key, val in state_dict.items():
            if 'Linear' in key or 'linear' in key:
                if 'weight' in key:
                    linear_weights['weight'] = val
                elif 'bias' in key:
                    linear_weights['bias'] = val
        
        if linear_weights:
            self.base_model.set_weights(linear_weights)
            self.is_loaded = True
        
        return self
    
    def load_from_file(self, filepath: str) -> 'NLinearForecaster':
        """从文件加载模型"""
        try:
            weights = DLinearDataLoader.load_from_file(filepath)
            
            # Try to find linear weights
            linear_weights = {}
            for key, val in weights.items():
                if 'Linear' in key or 'linear' in key:
                    if 'weight' in key:
                        linear_weights['weight'] = val
                    elif 'bias' in key:
                        linear_weights['bias'] = val
            
            if linear_weights:
                self.base_model.set_weights(linear_weights)
            
            self.is_loaded = True
            return self
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {str(e)}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """进行预测"""
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction.")
        
        # Normalize input shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, n_features = x.shape
        
        if seq_len != self.seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) does not match expected ({self.seq_len})")
        
        predictions = self.base_model.forward(x)
        
        if batch_size == 1 and n_features == 1:
            predictions = predictions.squeeze()
        elif batch_size == 1:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self.base_model.get_model_info()
        info['is_loaded'] = self.is_loaded
        info['task'] = 'forecasting'
        return info


class AutoformerForecaster:
    """
    Autoformer预测模型
    
    Forecasting model using Autoformer base model with auto-correlation.
    """
    
    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        enc_in: int = 1,
        dec_in: int = 1,
        c_out: int = 1,
        kernel_size: int = 25,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        label_len : int
            标签长度 / Label length
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
            前馈网络维度 / Feed-forward dimension
        enc_in : int, default=1
            编码器输入维度 / Encoder input dimension
        dec_in : int, default=1
            解码器输入维度 / Decoder input dimension
        c_out : int, default=1
            输出维度 / Output dimension
        kernel_size : int, default=25
            分解核大小 / Decomposition kernel size
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.base_model = AutoformerBaseModel(
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.is_loaded = False
    
    @property
    def seq_len(self):
        return self.base_model.seq_len
    
    @property
    def pred_len(self):
        return self.base_model.pred_len
    
    @property
    def label_len(self):
        return self.base_model.label_len
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'AutoformerForecaster':
        """从字典加载模型参数"""
        enc_embedding = state_dict.get('enc_embedding')
        dec_embedding = state_dict.get('dec_embedding')
        projection = state_dict.get('projection')
        
        self.base_model.set_weights(enc_embedding, dec_embedding, projection)
        self.is_loaded = True
        return self
    
    def predict(
        self,
        x_enc: np.ndarray,
        x_dec: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """进行预测"""
        # Normalize input shape
        if x_enc.ndim == 1:
            x_enc = x_enc.reshape(1, -1, 1)
        elif x_enc.ndim == 2:
            x_enc = x_enc.reshape(1, x_enc.shape[0], x_enc.shape[1])
        
        batch_size, seq_len, n_features = x_enc.shape
        
        predictions = self.base_model.forward(x_enc, x_dec)
        
        if batch_size == 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze()
        elif batch_size == 1:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self.base_model.get_model_info()
        info['is_loaded'] = self.is_loaded
        info['task'] = 'forecasting'
        return info


class TimeXerForecaster:
    """
    TimeXer预测模型
    
    Forecasting model using TimeXer base model with exogenous variables support.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        enc_in: int = 1,
        n_exog: int = 0,
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
        patch_len : int, default=16
            Patch长度 / Patch length
        stride : int, default=8
            步长 / Stride
        d_model : int, default=512
            模型维度 / Model dimension
        n_heads : int, default=8
            注意力头数 / Number of attention heads
        e_layers : int, default=2
            编码器层数 / Number of encoder layers
        d_ff : int, default=2048
            前馈网络维度 / Feed-forward dimension
        enc_in : int, default=1
            编码器输入维度 / Encoder input dimension
        n_exog : int, default=0
            外生变量数量 / Number of exogenous variables
        c_out : int, default=1
            输出维度 / Output dimension
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.base_model = TimeXerBaseModel(
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            enc_in=enc_in,
            n_exog=n_exog,
            c_out=c_out,
            dropout=dropout
        )
        self.is_loaded = False
    
    @property
    def seq_len(self):
        return self.base_model.seq_len
    
    @property
    def pred_len(self):
        return self.base_model.pred_len
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'TimeXerForecaster':
        """从字典加载模型参数"""
        patch_embedding = state_dict.get('patch_embedding')
        exog_embedding = state_dict.get('exog_embedding')
        prediction_head = state_dict.get('prediction_head')
        global_token = state_dict.get('global_token')
        
        self.base_model.set_weights(
            patch_embedding, exog_embedding, prediction_head, global_token
        )
        self.is_loaded = True
        return self
    
    def predict(
        self,
        x_endog: np.ndarray,
        x_exog: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """进行预测"""
        # Normalize input shape
        if x_endog.ndim == 1:
            x_endog = x_endog.reshape(1, -1, 1)
        elif x_endog.ndim == 2:
            x_endog = x_endog.reshape(1, x_endog.shape[0], x_endog.shape[1])
        
        batch_size = x_endog.shape[0]
        
        predictions = self.base_model.forward(x_endog, x_exog)
        
        if batch_size == 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze()
        elif batch_size == 1:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self.base_model.get_model_info()
        info['is_loaded'] = self.is_loaded
        info['task'] = 'forecasting'
        return info


class TimeMixerForecaster:
    """
    TimeMixer预测模型
    
    Forecasting model using TimeMixer base model with multi-scale mixing.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        d_ff: int = 2048,
        e_layers: int = 2,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
        enc_in: int = 1,
        c_out: int = 1,
        kernel_size: int = 25,
        dropout: float = 0.1,
        use_revin: bool = True
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
        d_ff : int, default=2048
            前馈网络维度 / Feed-forward dimension
        e_layers : int, default=2
            编码器层数 / Number of encoder layers
        down_sampling_layers : int, default=3
            下采样层数 / Number of downsampling layers
        down_sampling_window : int, default=2
            下采样窗口大小 / Downsampling window size
        enc_in : int, default=1
            编码器输入维度 / Encoder input dimension
        c_out : int, default=1
            输出维度 / Output dimension
        kernel_size : int, default=25
            分解核大小 / Decomposition kernel size
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        use_revin : bool, default=True
            是否使用RevIN / Whether to use RevIN
        """
        self.base_model = TimeMixerBaseModel(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            down_sampling_layers=down_sampling_layers,
            down_sampling_window=down_sampling_window,
            enc_in=enc_in,
            c_out=c_out,
            kernel_size=kernel_size,
            dropout=dropout,
            use_revin=use_revin
        )
        self.is_loaded = False
    
    @property
    def seq_len(self):
        return self.base_model.seq_len
    
    @property
    def pred_len(self):
        return self.base_model.pred_len
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'TimeMixerForecaster':
        """从字典加载模型参数"""
        seasonal_prediction = state_dict.get('seasonal_prediction')
        trend_prediction = state_dict.get('trend_prediction')
        output_projection = state_dict.get('output_projection')
        revin_weights = state_dict.get('revin')
        
        self.base_model.set_weights(
            seasonal_prediction, trend_prediction, output_projection, revin_weights
        )
        self.is_loaded = True
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """进行预测"""
        # Normalize input shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size = x.shape[0]
        
        predictions = self.base_model.forward(x)
        
        if batch_size == 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze()
        elif batch_size == 1:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self.base_model.get_model_info()
        info['is_loaded'] = self.is_loaded
        info['task'] = 'forecasting'
        return info


class ShortTermPredictor:
    """
    短期预测器
    
    A unified interface for short-term parameter prediction using various models.
    Supports multiple underlying model architectures.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        model_type: str = 'dlinear',
        enc_in: int = 1,
        **kwargs
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        pred_len : int
            预测长度 / Prediction length
        model_type : str, default='dlinear'
            模型类型 ('dlinear', 'nlinear', 'transformer', 'autoformer', 'timexer', 'timemixer')
        enc_in : int, default=1
            输入特征维度 / Number of input features
        **kwargs : dict
            模型特定参数 / Model-specific parameters
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_type = model_type
        self.enc_in = enc_in
        
        # Import here to avoid circular imports
        from .forecasting import DLinearForecaster, TransformerForecaster
        
        # Create appropriate model
        if model_type == 'dlinear':
            self.model = DLinearForecaster(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                **kwargs
            )
        elif model_type == 'nlinear':
            self.model = NLinearForecaster(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                **kwargs
            )
        elif model_type == 'transformer':
            self.model = TransformerForecaster(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                **kwargs
            )
        elif model_type == 'autoformer':
            label_len = kwargs.pop('label_len', seq_len // 2)
            self.model = AutoformerForecaster(
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                enc_in=enc_in,
                **kwargs
            )
        elif model_type == 'timexer':
            self.model = TimeXerForecaster(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                **kwargs
            )
        elif model_type == 'timemixer':
            self.model = TimeMixerForecaster(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @property
    def is_loaded(self):
        return self.model.is_loaded
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'ShortTermPredictor':
        """从字典加载模型参数"""
        self.model.load_from_dict(state_dict)
        return self
    
    def load_from_file(self, filepath: str) -> 'ShortTermPredictor':
        """从文件加载模型"""
        if hasattr(self.model, 'load_from_file'):
            self.model.load_from_file(filepath)
        else:
            # Try using DLinearDataLoader
            weights = DLinearDataLoader.load_from_file(filepath)
            self.model.load_from_dict(weights)
        return self
    
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        进行短期预测
        Make short-term prediction
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列
        **kwargs : dict
            模型特定参数（如外生变量等）
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果
        """
        return self.model.predict(x, **kwargs)
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Make the model callable"""
        return self.predict(x, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self.model.get_model_info()
        info['wrapper'] = 'ShortTermPredictor'
        return info
