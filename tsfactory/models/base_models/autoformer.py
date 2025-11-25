"""
Autoformer Base Model

Autoformer基础模型，使用自相关机制进行时序预测
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from .layers import SeriesDecomp, AutoCorrelation, AutoCorrelationLayer, LayerNorm, FeedForward


class AutoformerEncoderLayer:
    """
    Autoformer编码器层
    
    Encoder layer with auto-correlation and series decomposition.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        kernel_size: int = 25,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        n_heads : int
            注意力头数 / Number of attention heads
        d_ff : int
            前馈网络维度 / Feed-forward dimension
        kernel_size : int, default=25
            分解核大小 / Decomposition kernel size
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        self.autocorrelation = AutoCorrelation(d_model, n_heads, attention_dropout=dropout)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.feedforward = FeedForward(d_model, d_ff, dropout=dropout, activation='relu')
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch, seq_len, d_model)
            
        Returns:
        --------
        seasonal : np.ndarray
            季节性输出
        trend : np.ndarray
            趋势输出
        """
        # Auto-correlation
        attn_out, _ = self.autocorrelation(x, x, x)
        x = x + attn_out
        
        # First decomposition
        seasonal, trend = self.decomp1(x)
        
        # Feed-forward
        ff_out = self.feedforward(seasonal)
        seasonal = seasonal + ff_out
        
        # Second decomposition
        seasonal, trend2 = self.decomp2(seasonal)
        trend = trend + trend2
        
        return seasonal, trend
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make the layer callable"""
        return self.forward(x)


class AutoformerDecoderLayer:
    """
    Autoformer解码器层
    
    Decoder layer with auto-correlation, cross-attention, and series decomposition.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        kernel_size: int = 25,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        n_heads : int
            注意力头数 / Number of attention heads
        d_ff : int
            前馈网络维度 / Feed-forward dimension
        kernel_size : int, default=25
            分解核大小 / Decomposition kernel size
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        self.self_attention = AutoCorrelation(d_model, n_heads, attention_dropout=dropout)
        self.cross_attention = AutoCorrelation(d_model, n_heads, attention_dropout=dropout)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.decomp3 = SeriesDecomp(kernel_size)
        self.feedforward = FeedForward(d_model, d_ff, dropout=dropout, activation='relu')
        
        # Trend projections
        self.trend_projection = None
    
    def forward(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        trend_init: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            解码器输入，shape为 (batch, seq_len, d_model)
        encoder_output : np.ndarray
            编码器输出，shape为 (batch, seq_len, d_model)
        trend_init : np.ndarray
            趋势初始值
            
        Returns:
        --------
        seasonal : np.ndarray
            季节性输出
        trend : np.ndarray
            趋势输出
        """
        # Self-attention
        self_attn_out, _ = self.self_attention(x, x, x)
        x = x + self_attn_out
        seasonal, trend1 = self.decomp1(x)
        
        # Cross-attention
        cross_attn_out, _ = self.cross_attention(seasonal, encoder_output, encoder_output)
        seasonal = seasonal + cross_attn_out
        seasonal, trend2 = self.decomp2(seasonal)
        
        # Feed-forward
        ff_out = self.feedforward(seasonal)
        seasonal = seasonal + ff_out
        seasonal, trend3 = self.decomp3(seasonal)
        
        # Aggregate trends
        trend = trend_init + trend1 + trend2 + trend3
        
        return seasonal, trend
    
    def __call__(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        trend_init: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make the layer callable"""
        return self.forward(x, encoder_output, trend_init)


class AutoformerBaseModel:
    """
    Autoformer基础模型
    
    Autoformer is a transformer-based model that decomposes time series
    into trend and seasonal components and uses auto-correlation mechanism
    for capturing temporal dependencies.
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
            标签长度（用于解码器输入）/ Label length for decoder input
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
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        # Series decomposition
        self.decomposition = SeriesDecomp(kernel_size)
        
        # Encoder layers
        self.encoder_layers = [
            AutoformerEncoderLayer(d_model, n_heads, d_ff, kernel_size, dropout)
            for _ in range(e_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            AutoformerDecoderLayer(d_model, n_heads, d_ff, kernel_size, dropout)
            for _ in range(d_layers)
        ]
        
        # Projections
        self.enc_embedding = None
        self.dec_embedding = None
        self.projection = None
    
    def set_weights(
        self,
        enc_embedding: Optional[Dict[str, np.ndarray]] = None,
        dec_embedding: Optional[Dict[str, np.ndarray]] = None,
        projection: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        设置模型权重
        Set model weights
        """
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.projection = projection
    
    def _embed(self, x: np.ndarray, embedding: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """应用嵌入层"""
        if embedding is None:
            # Simple projection to d_model
            batch, seq_len, n_features = x.shape
            if n_features < self.d_model:
                output = np.zeros((batch, seq_len, self.d_model))
                output[:, :, :n_features] = x
                return output
            else:
                return x[:, :, :self.d_model]
        
        # Apply embedding weights
        weight = embedding.get('weight')
        if weight is not None:
            return x @ weight.T
        return x
    
    def _project(self, x: np.ndarray) -> np.ndarray:
        """应用输出投影"""
        if self.projection is None:
            return x[:, :, :self.c_out]
        
        weight = self.projection.get('weight')
        bias = self.projection.get('bias')
        
        output = x @ weight.T if weight is not None else x
        if bias is not None:
            output = output + bias
        return output
    
    def forward(self, x_enc: np.ndarray, x_dec: Optional[np.ndarray] = None) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x_enc : np.ndarray
            编码器输入，shape为 (batch, seq_len, enc_in)
        x_dec : np.ndarray, optional
            解码器输入，shape为 (batch, label_len + pred_len, dec_in)
            如果未提供，将自动构建
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果，shape为 (batch, pred_len, c_out)
        """
        batch_size = x_enc.shape[0]
        
        # Decompose input for initialization
        seasonal_init, trend_init = self.decomposition(x_enc)
        
        # Embed encoder input
        enc_out = self._embed(x_enc, self.enc_embedding)
        
        # Encoder
        for enc_layer in self.encoder_layers:
            enc_out, _ = enc_layer(enc_out)
        
        # Prepare decoder input
        if x_dec is None:
            # Use mean as decoder seasonal input
            dec_seasonal = np.mean(seasonal_init, axis=1, keepdims=True)
            dec_seasonal = np.repeat(dec_seasonal, self.label_len + self.pred_len, axis=1)
            # Use last trend value extended
            dec_trend = trend_init[:, -1:, :]
            dec_trend = np.repeat(dec_trend, self.label_len + self.pred_len, axis=1)
        else:
            dec_seasonal, dec_trend = self.decomposition(x_dec)
        
        # Embed decoder input
        dec_out = self._embed(dec_seasonal, self.dec_embedding)
        
        # Decoder
        for dec_layer in self.decoder_layers:
            dec_out, dec_trend = dec_layer(dec_out, enc_out, dec_trend)
        
        # Combine seasonal and trend
        output = dec_out + dec_trend
        
        # Project to output dimension and take prediction part
        output = self._project(output)
        predictions = output[:, -self.pred_len:, :]
        
        return predictions
    
    def __call__(self, x_enc: np.ndarray, x_dec: Optional[np.ndarray] = None) -> np.ndarray:
        """Make the model callable"""
        return self.forward(x_enc, x_dec)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        Get model information
        """
        return {
            'model_type': 'Autoformer',
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'e_layers': self.e_layers,
            'd_layers': self.d_layers,
            'd_ff': self.d_ff,
            'enc_in': self.enc_in,
            'dec_in': self.dec_in,
            'c_out': self.c_out,
            'kernel_size': self.kernel_size,
            'has_enc_embedding': self.enc_embedding is not None,
            'has_dec_embedding': self.dec_embedding is not None,
            'has_projection': self.projection is not None
        }
