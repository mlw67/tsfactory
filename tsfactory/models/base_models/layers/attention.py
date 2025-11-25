"""
Attention Layers

注意力层，用于Transformer类模型
"""

import numpy as np
from typing import Tuple, Optional


class FullAttention:
    """
    全注意力层
    
    Standard multi-head self-attention mechanism.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
        scale: Optional[float] = None
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        n_heads : int
            注意力头数 / Number of attention heads
        attention_dropout : float, default=0.1
            注意力Dropout率 / Attention dropout rate
        output_attention : bool, default=False
            是否输出注意力权重 / Whether to output attention weights
        scale : float, optional
            缩放因子 / Scale factor
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.scale = scale or (1.0 / np.sqrt(self.d_keys))
        
        # Weights
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
        """设置权重"""
        self.query_projection = query_projection
        self.key_projection = key_projection
        self.value_projection = value_projection
        self.out_projection = out_projection
    
    def _linear_transform(self, x: np.ndarray, weights: dict) -> np.ndarray:
        """线性变换"""
        if weights is None:
            return x
        
        output = x @ weights.get('weight', np.eye(x.shape[-1])).T
        if 'bias' in weights:
            output = output + weights['bias']
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax函数"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)
    
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
            输出
        attn_weights : np.ndarray, optional
            注意力权重
        """
        batch, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
        
        # Linear projections
        queries = self._linear_transform(queries, self.query_projection)
        keys = self._linear_transform(keys, self.key_projection)
        values = self._linear_transform(values, self.value_projection)
        
        # Reshape to (batch, n_heads, seq_len, d_keys)
        queries = queries.reshape(batch, q_len, self.n_heads, self.d_keys)
        queries = np.transpose(queries, (0, 2, 1, 3))
        
        keys = keys.reshape(batch, k_len, self.n_heads, self.d_keys)
        keys = np.transpose(keys, (0, 2, 1, 3))
        
        values = values.reshape(batch, k_len, self.n_heads, self.d_keys)
        values = np.transpose(values, (0, 2, 1, 3))
        
        # Attention scores
        scores = np.matmul(queries, np.transpose(keys, (0, 1, 3, 2))) * self.scale
        
        # Apply mask
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Softmax
        attn_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        output = np.matmul(attn_weights, values)
        
        # Reshape back
        output = np.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, q_len, self.d_model)
        
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


class ProbAttention:
    """
    ProbSparse注意力层
    
    Probability sparse attention for Informer model.
    Selects dominant queries based on sparsity measurement.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 5,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
        scale: Optional[float] = None
    ):
        """
        Parameters:
        -----------
        d_model : int
            模型维度 / Model dimension
        n_heads : int
            注意力头数 / Number of attention heads
        factor : int, default=5
            采样因子 / Sampling factor
        attention_dropout : float, default=0.1
            注意力Dropout率 / Attention dropout rate
        output_attention : bool, default=False
            是否输出注意力权重 / Whether to output attention weights
        scale : float, optional
            缩放因子 / Scale factor
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.scale = scale or (1.0 / np.sqrt(self.d_keys))
        
        # Weights
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
        """设置权重"""
        self.query_projection = query_projection
        self.key_projection = key_projection
        self.value_projection = value_projection
        self.out_projection = out_projection
    
    def _linear_transform(self, x: np.ndarray, weights: dict) -> np.ndarray:
        """线性变换"""
        if weights is None:
            return x
        
        output = x @ weights.get('weight', np.eye(x.shape[-1])).T
        if 'bias' in weights:
            output = output + weights['bias']
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax函数"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)
    
    def _prob_QK(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        sample_k: int,
        n_top: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算稀疏查询选择
        Calculate sparse query selection
        """
        batch, n_heads, q_len, d_keys = Q.shape
        _, _, k_len, _ = K.shape
        
        # Sample keys
        K_sample = K[:, :, np.random.choice(k_len, sample_k, replace=False), :]
        
        # Calculate Q_K
        Q_K_sample = np.matmul(Q, np.transpose(K_sample, (0, 1, 3, 2)))
        
        # Find sparsity measurement
        M = np.max(Q_K_sample, axis=-1) - np.mean(Q_K_sample, axis=-1)
        
        # Select top-n queries
        M_top_indices = np.argsort(M, axis=-1)[:, :, -n_top:]
        
        # Get top queries
        Q_top = np.zeros((batch, n_heads, n_top, d_keys))
        for b in range(batch):
            for h in range(n_heads):
                Q_top[b, h] = Q[b, h, M_top_indices[b, h]]
        
        return Q_top, M_top_indices
    
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
        """
        batch, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
        
        # Linear projections
        queries = self._linear_transform(queries, self.query_projection)
        keys = self._linear_transform(keys, self.key_projection)
        values = self._linear_transform(values, self.value_projection)
        
        # Reshape to (batch, n_heads, seq_len, d_keys)
        queries = queries.reshape(batch, q_len, self.n_heads, self.d_keys)
        queries = np.transpose(queries, (0, 2, 1, 3))
        
        keys = keys.reshape(batch, k_len, self.n_heads, self.d_keys)
        keys = np.transpose(keys, (0, 2, 1, 3))
        
        values = values.reshape(batch, k_len, self.n_heads, self.d_keys)
        values = np.transpose(values, (0, 2, 1, 3))
        
        # ProbSparse attention
        U_part = self.factor * int(np.ceil(np.log(k_len)))
        u = self.factor * int(np.ceil(np.log(q_len)))
        
        U_part = min(U_part, k_len)
        u = min(u, q_len)
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        # Full attention on top queries
        scale = self.scale
        scores = np.matmul(scores_top, np.transpose(keys, (0, 1, 3, 2))) * scale
        
        if attn_mask is not None:
            # Apply mask to selected positions
            pass  # Simplified for now
        
        attn = self._softmax(scores, axis=-1)
        
        # Get context
        context = np.matmul(attn, values)
        
        # Initialize output with mean values
        output = np.mean(values, axis=2, keepdims=True)
        output = np.repeat(output, q_len, axis=2)
        
        # Fill in top query results
        for b in range(batch):
            for h in range(self.n_heads):
                for i, idx in enumerate(index[b, h]):
                    output[b, h, idx] = context[b, h, i]
        
        # Reshape back
        output = np.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, q_len, self.d_model)
        
        # Output projection
        output = self._linear_transform(output, self.out_projection)
        
        if self.output_attention:
            return output, attn
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


class AttentionLayer:
    """
    注意力层封装
    
    Wrapper for attention mechanism with residual connection.
    """
    
    def __init__(
        self,
        attention,
        d_model: int,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        attention : FullAttention or ProbAttention
            注意力模块
        d_model : int
            模型维度
        dropout : float, default=0.1
            Dropout率
        """
        self.attention = attention
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
        
        output, attn = self.attention(x, cross, cross, attn_mask)
        
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


class CrossAttention:
    """
    交叉注意力层
    
    Cross-attention between two sequences, used in encoder-decoder architectures.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
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
        attention_dropout : float, default=0.1
            注意力Dropout率 / Attention dropout rate
        output_attention : bool, default=False
            是否输出注意力权重 / Whether to output attention weights
        """
        self.attention = FullAttention(
            d_model=d_model,
            n_heads=n_heads,
            attention_dropout=attention_dropout,
            output_attention=output_attention
        )
    
    def set_weights(
        self,
        query_projection: Optional[dict] = None,
        key_projection: Optional[dict] = None,
        value_projection: Optional[dict] = None,
        out_projection: Optional[dict] = None
    ):
        """设置权重"""
        self.attention.set_weights(
            query_projection, key_projection, value_projection, out_projection
        )
    
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
        """
        return self.attention.forward(queries, keys, values, attn_mask)
    
    def __call__(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        attn_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make the layer callable"""
        return self.forward(queries, keys, values, attn_mask)
