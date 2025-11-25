"""
Autocorrelation Layer

自相关层，用于Autoformer模型
"""

import numpy as np
from typing import Tuple, Optional


class AutoCorrelation:
    """
    自相关层
    
    Autocorrelation layer for capturing temporal dependencies.
    Uses frequency domain operations for efficient auto-correlation computation.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 3,
        attention_dropout: float = 0.1,
        output_attention: bool = False
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        n_heads : int
            注意力头数 / Number of attention heads
        factor : int, default=3
            用于选择top-k自相关的因子 / Factor for selecting top-k autocorrelation
        attention_dropout : float, default=0.1
            注意力Dropout率 / Attention dropout rate
        output_attention : bool, default=False
            是否输出注意力权重 / Whether to output attention weights
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        
        # Weights (will be set when loading)
        self.query_projection = None
        self.key_projection = None
        self.value_projection = None
        self.out_projection = None
    
    def set_weights(
        self,
        query_projection: Optional[dict] = None,
        key_projection: Optional[dict] = None,
        value_projection: Optional[dict] = None,
        out_projection: Optional[dict] = None
    ):
        """
        设置层权重
        Set layer weights
        """
        self.query_projection = query_projection
        self.key_projection = key_projection
        self.value_projection = value_projection
        self.out_projection = out_projection
    
    def _time_delay_agg_full(
        self,
        values: np.ndarray,
        corr: np.ndarray
    ) -> np.ndarray:
        """
        时间延迟聚合（完整版本）
        Time delay aggregation (full version)
        
        Parameters:
        -----------
        values : np.ndarray
            值张量，shape为 (batch, heads, length, d_keys)
        corr : np.ndarray
            相关系数，shape为 (batch, heads, length)
            
        Returns:
        --------
        output : np.ndarray
            聚合后的输出
        """
        batch, heads, length, d_keys = values.shape
        
        # Get top-k delays
        top_k = max(int(self.factor * np.log(length)), 1)
        top_k = min(top_k, length)
        
        # Get indices of top-k correlations
        top_k_indices = np.argsort(corr, axis=-1)[:, :, -top_k:]
        top_k_weights = np.take_along_axis(corr, top_k_indices, axis=-1)
        
        # Softmax over selected weights
        top_k_weights_max = np.max(top_k_weights, axis=-1, keepdims=True)
        top_k_weights = np.exp(top_k_weights - top_k_weights_max)
        top_k_weights = top_k_weights / (np.sum(top_k_weights, axis=-1, keepdims=True) + 1e-8)
        
        # Aggregate values with delays
        output = np.zeros_like(values)
        for i in range(top_k):
            delay = top_k_indices[:, :, i:i+1]  # (batch, heads, 1)
            weight = top_k_weights[:, :, i:i+1]  # (batch, heads, 1)
            
            # Roll values by delay and weight
            for b in range(batch):
                for h in range(heads):
                    d = int(delay[b, h, 0])
                    rolled = np.roll(values[b, h], shift=d, axis=0)
                    output[b, h] += weight[b, h] * rolled
        
        return output
    
    def _auto_correlation(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        自相关计算
        Autocorrelation computation
        
        Parameters:
        -----------
        queries : np.ndarray
            查询张量，shape为 (batch, heads, seq_len, d_keys)
        keys : np.ndarray
            键张量，shape为 (batch, heads, seq_len, d_keys)
        values : np.ndarray
            值张量，shape为 (batch, heads, seq_len, d_keys)
            
        Returns:
        --------
        output : np.ndarray
            自相关输出
        attn_weights : np.ndarray
            注意力权重
        """
        batch, heads, length, d_keys = queries.shape
        
        # Compute autocorrelation using FFT
        q_fft = np.fft.rfft(queries, axis=2)
        k_fft = np.fft.rfft(keys, axis=2)
        
        # Cross-correlation in frequency domain
        res = q_fft * np.conj(k_fft)
        
        # Convert back to time domain
        corr = np.fft.irfft(res, n=length, axis=2)
        
        # Average correlation across d_keys dimension
        corr = np.mean(corr, axis=-1)  # (batch, heads, length)
        
        # Aggregate values with time delays
        output = self._time_delay_agg_full(values, corr)
        
        return output, corr
    
    def _linear_transform(
        self,
        x: np.ndarray,
        weights: dict
    ) -> np.ndarray:
        """线性变换"""
        if weights is None:
            return x
            
        output = x @ weights.get('weight', np.eye(x.shape[-1])).T
        if 'bias' in weights:
            output = output + weights['bias']
        return output
    
    def forward(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        attn_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        queries : np.ndarray
            查询，shape为 (batch, seq_len, d_model)
        keys : np.ndarray
            键，shape为 (batch, seq_len, d_model)
        values : np.ndarray
            值，shape为 (batch, seq_len, d_model)
        attn_mask : np.ndarray, optional
            注意力掩码
            
        Returns:
        --------
        output : np.ndarray
            输出，shape为 (batch, seq_len, d_model)
        attn_weights : np.ndarray, optional
            注意力权重（如果output_attention=True）
        """
        batch, seq_len, _ = queries.shape
        
        # Linear projections
        queries = self._linear_transform(queries, self.query_projection)
        keys = self._linear_transform(keys, self.key_projection)
        values = self._linear_transform(values, self.value_projection)
        
        # Reshape to (batch, heads, seq_len, d_keys)
        queries = queries.reshape(batch, seq_len, self.n_heads, self.d_keys)
        queries = np.transpose(queries, (0, 2, 1, 3))
        
        keys = keys.reshape(batch, seq_len, self.n_heads, self.d_keys)
        keys = np.transpose(keys, (0, 2, 1, 3))
        
        values = values.reshape(batch, seq_len, self.n_heads, self.d_keys)
        values = np.transpose(values, (0, 2, 1, 3))
        
        # Auto-correlation
        output, attn_weights = self._auto_correlation(queries, keys, values)
        
        # Reshape back to (batch, seq_len, d_model)
        output = np.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.d_model)
        
        # Output projection
        output = self._linear_transform(output, self.out_projection)
        
        if self.output_attention:
            return output, attn_weights
        return output, None
    
    def __call__(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        attn_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make the layer callable"""
        return self.forward(queries, keys, values, attn_mask)


class AutoCorrelationLayer:
    """
    自相关层封装
    
    Wrapper for AutoCorrelation with residual connection and normalization.
    """
    
    def __init__(
        self,
        autocorrelation: AutoCorrelation,
        d_model: int,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        autocorrelation : AutoCorrelation
            自相关模块
        d_model : int
            模型维度
        dropout : float, default=0.1
            Dropout率
        """
        self.autocorrelation = autocorrelation
        self.d_model = d_model
        self.dropout = dropout
    
    def forward(
        self,
        x: np.ndarray,
        cross: Optional[np.ndarray] = None,
        attn_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        前向传播
        Forward pass
        """
        if cross is None:
            cross = x
        
        output, attn = self.autocorrelation(x, cross, cross, attn_mask)
        
        # Residual connection
        output = x + output
        
        return output, attn
    
    def __call__(
        self,
        x: np.ndarray,
        cross: Optional[np.ndarray] = None,
        attn_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make the layer callable"""
        return self.forward(x, cross, attn_mask)
