"""
TimeMixer Base Model

TimeMixer基础模型，使用多尺度混合进行时序预测
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .layers import SeriesDecomp, FeedForward, LayerNorm, RevIN


class MultiScaleSeasonMixing:
    """
    多尺度季节性混合
    
    Mixes seasonal components across different scales.
    """
    
    def __init__(
        self,
        seq_len: int,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
        d_model: int = 512,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        down_sampling_layers : int, default=3
            下采样层数 / Number of downsampling layers
        down_sampling_window : int, default=2
            下采样窗口大小 / Downsampling window size
        d_model : int, default=512
            模型维度 / Model dimension
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.seq_len = seq_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.d_model = d_model
        
        # Calculate sequence lengths at each scale
        self.scale_lengths = [seq_len]
        for i in range(down_sampling_layers):
            self.scale_lengths.append(self.scale_lengths[-1] // down_sampling_window)
        
        # Mixing weights
        self.mixing_weights = None
    
    def set_weights(self, mixing_weights: Optional[Dict[str, np.ndarray]] = None):
        """设置权重"""
        self.mixing_weights = mixing_weights
    
    def _downsample(self, x: np.ndarray, window: int) -> np.ndarray:
        """下采样"""
        batch, seq_len, n_features = x.shape
        new_len = seq_len // window
        x_reshaped = x[:, :new_len * window, :].reshape(batch, new_len, window, n_features)
        return np.mean(x_reshaped, axis=2)
    
    def _upsample(self, x: np.ndarray, target_len: int) -> np.ndarray:
        """上采样"""
        batch, seq_len, n_features = x.shape
        ratio = target_len // seq_len
        return np.repeat(x, ratio, axis=1)[:, :target_len, :]
    
    def forward(self, season_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        season_list : List[np.ndarray]
            多尺度季节性分量列表
            
        Returns:
        --------
        mixed_list : List[np.ndarray]
            混合后的季节性分量
        """
        n_scales = len(season_list)
        mixed_list = []
        
        for i in range(n_scales):
            mixed = np.zeros_like(season_list[i])
            
            for j in range(n_scales):
                if j < i:
                    # Upsample lower resolution
                    upsampled = self._upsample(season_list[j], season_list[i].shape[1])
                    mixed = mixed + upsampled
                elif j > i:
                    # Downsample higher resolution
                    target_len = season_list[i].shape[1]
                    src_len = season_list[j].shape[1]
                    ratio = src_len // target_len
                    if ratio > 0:
                        downsampled = self._downsample(season_list[j], ratio)
                        mixed = mixed + downsampled[:, :target_len, :]
                else:
                    mixed = mixed + season_list[j]
            
            mixed = mixed / n_scales
            mixed_list.append(mixed)
        
        return mixed_list
    
    def __call__(self, season_list: List[np.ndarray]) -> List[np.ndarray]:
        """Make the layer callable"""
        return self.forward(season_list)


class MultiScaleTrendMixing:
    """
    多尺度趋势混合
    
    Mixes trend components across different scales.
    """
    
    def __init__(
        self,
        seq_len: int,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
        d_model: int = 512,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        down_sampling_layers : int, default=3
            下采样层数 / Number of downsampling layers
        down_sampling_window : int, default=2
            下采样窗口大小 / Downsampling window size
        d_model : int, default=512
            模型维度 / Model dimension
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.seq_len = seq_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.d_model = d_model
        
        # Mixing weights
        self.mixing_weights = None
    
    def set_weights(self, mixing_weights: Optional[Dict[str, np.ndarray]] = None):
        """设置权重"""
        self.mixing_weights = mixing_weights
    
    def _downsample(self, x: np.ndarray, window: int) -> np.ndarray:
        """下采样"""
        batch, seq_len, n_features = x.shape
        new_len = seq_len // window
        if new_len == 0:
            return x
        x_reshaped = x[:, :new_len * window, :].reshape(batch, new_len, window, n_features)
        return np.mean(x_reshaped, axis=2)
    
    def _upsample(self, x: np.ndarray, target_len: int) -> np.ndarray:
        """上采样"""
        batch, seq_len, n_features = x.shape
        ratio = target_len // seq_len
        if ratio == 0:
            return x[:, :target_len, :]
        return np.repeat(x, ratio, axis=1)[:, :target_len, :]
    
    def forward(self, trend_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        前向传播
        Forward pass
        """
        n_scales = len(trend_list)
        mixed_list = []
        
        for i in range(n_scales):
            mixed = np.zeros_like(trend_list[i])
            count = 0
            
            for j in range(n_scales):
                if j < i:
                    upsampled = self._upsample(trend_list[j], trend_list[i].shape[1])
                    if upsampled.shape[1] == trend_list[i].shape[1]:
                        mixed = mixed + upsampled
                        count += 1
                elif j > i:
                    target_len = trend_list[i].shape[1]
                    src_len = trend_list[j].shape[1]
                    ratio = src_len // target_len
                    if ratio > 0:
                        downsampled = self._downsample(trend_list[j], ratio)
                        if downsampled.shape[1] >= target_len:
                            mixed = mixed + downsampled[:, :target_len, :]
                            count += 1
                else:
                    mixed = mixed + trend_list[j]
                    count += 1
            
            if count > 0:
                mixed = mixed / count
            mixed_list.append(mixed)
        
        return mixed_list
    
    def __call__(self, trend_list: List[np.ndarray]) -> List[np.ndarray]:
        """Make the layer callable"""
        return self.forward(trend_list)


class PastDecomposableMixing:
    """
    过去可分解混合层
    
    Decomposes and mixes past series at multiple scales.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
        d_model: int = 512,
        d_ff: int = 2048,
        kernel_size: int = 25,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        pred_len : int
            预测长度 / Prediction length
        down_sampling_layers : int, default=3
            下采样层数 / Number of downsampling layers
        down_sampling_window : int, default=2
            下采样窗口大小 / Downsampling window size
        d_model : int, default=512
            模型维度 / Model dimension
        d_ff : int, default=2048
            前馈网络维度 / Feed-forward dimension
        kernel_size : int, default=25
            分解核大小 / Decomposition kernel size
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        
        # Decomposition
        self.decomposition = SeriesDecomp(kernel_size)
        
        # Multi-scale mixing
        self.season_mixing = MultiScaleSeasonMixing(
            seq_len, down_sampling_layers, down_sampling_window, d_model, dropout
        )
        self.trend_mixing = MultiScaleTrendMixing(
            seq_len, down_sampling_layers, down_sampling_window, d_model, dropout
        )
    
    def _downsample(self, x: np.ndarray, window: int) -> np.ndarray:
        """下采样"""
        batch, seq_len, n_features = x.shape
        new_len = seq_len // window
        if new_len == 0:
            return x
        x_reshaped = x[:, :new_len * window, :].reshape(batch, new_len, window, n_features)
        return np.mean(x_reshaped, axis=2)
    
    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch, seq_len, n_features)
            
        Returns:
        --------
        season_list : List[np.ndarray]
            混合后的多尺度季节性分量
        trend_list : List[np.ndarray]
            混合后的多尺度趋势分量
        """
        # Create multi-scale representations
        x_list = [x]
        current = x
        for _ in range(self.down_sampling_layers):
            current = self._downsample(current, self.down_sampling_window)
            x_list.append(current)
        
        # Decompose at each scale
        season_list = []
        trend_list = []
        for x_scale in x_list:
            season, trend = self.decomposition(x_scale)
            season_list.append(season)
            trend_list.append(trend)
        
        # Mix across scales
        season_list = self.season_mixing(season_list)
        trend_list = self.trend_mixing(trend_list)
        
        return season_list, trend_list
    
    def __call__(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Make the layer callable"""
        return self.forward(x)


class TimeMixerBaseModel:
    """
    TimeMixer基础模型
    
    TimeMixer is a time series forecasting model that leverages multi-scale
    mixing for both seasonal and trend components. It effectively captures
    temporal patterns at different granularities.
    
    Key features:
    1. Multi-scale decomposition into trend and seasonal
    2. Scale-wise mixing of seasonal and trend components
    3. Future prediction using mixed representations
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
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.enc_in = enc_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.use_revin = use_revin
        
        # RevIN for normalization
        if use_revin:
            self.revin = RevIN(enc_in)
        else:
            self.revin = None
        
        # Past decomposable mixing layers
        self.past_mixing_layers = [
            PastDecomposableMixing(
                seq_len, pred_len, down_sampling_layers, down_sampling_window,
                d_model, d_ff, kernel_size, dropout
            )
            for _ in range(e_layers)
        ]
        
        # Prediction heads
        self.seasonal_prediction = None
        self.trend_prediction = None
        self.output_projection = None
    
    def set_weights(
        self,
        seasonal_prediction: Optional[Dict[str, np.ndarray]] = None,
        trend_prediction: Optional[Dict[str, np.ndarray]] = None,
        output_projection: Optional[Dict[str, np.ndarray]] = None,
        revin_weights: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        设置模型权重
        Set model weights
        """
        self.seasonal_prediction = seasonal_prediction
        self.trend_prediction = trend_prediction
        self.output_projection = output_projection
        
        if revin_weights is not None and self.revin is not None:
            weight = revin_weights.get('weight')
            bias = revin_weights.get('bias')
            if weight is not None and bias is not None:
                self.revin.set_weights(weight, bias)
    
    def _predict_component(
        self,
        x_list: List[np.ndarray],
        prediction_weights: Optional[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """预测单个分量"""
        # Use the finest scale for prediction
        x = x_list[0]
        batch_size, seq_len, n_features = x.shape
        
        if prediction_weights is not None:
            weight = prediction_weights.get('weight')
            bias = prediction_weights.get('bias')
            
            # Flatten and project
            x_flat = x.reshape(batch_size, -1)
            
            if weight is not None:
                if weight.shape[1] <= x_flat.shape[1]:
                    output = x_flat[:, :weight.shape[1]] @ weight.T
                else:
                    padded = np.zeros((batch_size, weight.shape[1]))
                    padded[:, :x_flat.shape[1]] = x_flat
                    output = padded @ weight.T
                
                if bias is not None:
                    output = output + bias
                    
                return output.reshape(batch_size, self.pred_len, self.c_out)
        
        # Simple last value repetition as fallback
        last_value = x[:, -1:, :self.c_out]
        return np.repeat(last_value, self.pred_len, axis=1)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch, seq_len, enc_in)
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果，shape为 (batch, pred_len, c_out)
        """
        # Apply RevIN normalization
        if self.revin is not None:
            x = self.revin(x, mode='norm')
        
        # Apply past decomposable mixing layers
        for mixing_layer in self.past_mixing_layers:
            season_list, trend_list = mixing_layer(x)
            # Update x with mixed components for next layer
            x = season_list[0] + trend_list[0]
        
        # Predict future seasonal component
        seasonal_pred = self._predict_component(season_list, self.seasonal_prediction)
        
        # Predict future trend component
        trend_pred = self._predict_component(trend_list, self.trend_prediction)
        
        # Combine predictions
        predictions = seasonal_pred + trend_pred
        
        # Apply output projection if available
        if self.output_projection is not None:
            weight = self.output_projection.get('weight')
            bias = self.output_projection.get('bias')
            if weight is not None:
                batch_size = predictions.shape[0]
                pred_flat = predictions.reshape(batch_size, -1)
                if weight.shape[1] == pred_flat.shape[1]:
                    predictions = pred_flat @ weight.T
                    if bias is not None:
                        predictions = predictions + bias
                    predictions = predictions.reshape(batch_size, self.pred_len, self.c_out)
        
        # Apply RevIN denormalization
        if self.revin is not None:
            predictions = self.revin(predictions, mode='denorm')
        
        return predictions
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the model callable"""
        return self.forward(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        Get model information
        """
        return {
            'model_type': 'TimeMixer',
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'e_layers': self.e_layers,
            'down_sampling_layers': self.down_sampling_layers,
            'down_sampling_window': self.down_sampling_window,
            'enc_in': self.enc_in,
            'c_out': self.c_out,
            'kernel_size': self.kernel_size,
            'use_revin': self.use_revin,
            'has_seasonal_prediction': self.seasonal_prediction is not None,
            'has_trend_prediction': self.trend_prediction is not None,
            'has_output_projection': self.output_projection is not None
        }
