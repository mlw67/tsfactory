"""
TimeXer Base Model

TimeXer基础模型，使用外生变量进行时序预测
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from .layers import PatchEmbedding, FullAttention, AttentionLayer, FeedForward, LayerNorm


class TimeXerEncoderLayer:
    """
    TimeXer编码器层
    
    Encoder layer with patch-wise attention for endogenous variables
    and cross-attention with exogenous variables.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
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
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Self-attention for patches
        self.self_attention = FullAttention(d_model, n_heads, attention_dropout=dropout)
        
        # Cross-attention with exogenous variables
        self.cross_attention = FullAttention(d_model, n_heads, attention_dropout=dropout)
        
        # Feed-forward network
        self.feedforward = FeedForward(d_model, d_ff, dropout=dropout, activation='gelu')
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(
        self,
        x: np.ndarray,
        exog: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            内生变量patches，shape为 (batch, n_patches, d_model)
        exog : np.ndarray, optional
            外生变量，shape为 (batch, seq_len, d_model)
            
        Returns:
        --------
        output : np.ndarray
            输出
        """
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Cross-attention with exogenous variables
        if exog is not None:
            cross_out, _ = self.cross_attention(x, exog, exog)
            x = self.norm2(x + cross_out)
        
        # Feed-forward
        ff_out = self.feedforward(x)
        x = self.norm3(x + ff_out)
        
        return x
    
    def __call__(
        self,
        x: np.ndarray,
        exog: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x, exog)


class TimeXerBaseModel:
    """
    TimeXer基础模型
    
    TimeXer is a time series forecasting model that effectively utilizes
    exogenous variables through a patch-based approach and cross-attention.
    
    The model:
    1. Patches endogenous (target) variable
    2. Uses cross-attention to incorporate exogenous variables
    3. Predicts future values using learned representations
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
            步长 / Stride for patching
        d_model : int, default=512
            模型维度 / Model dimension
        n_heads : int, default=8
            注意力头数 / Number of attention heads
        e_layers : int, default=2
            编码器层数 / Number of encoder layers
        d_ff : int, default=2048
            前馈网络维度 / Feed-forward dimension
        enc_in : int, default=1
            编码器输入维度（内生变量数）/ Encoder input dimension
        n_exog : int, default=0
            外生变量数量 / Number of exogenous variables
        c_out : int, default=1
            输出维度 / Output dimension
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.enc_in = enc_in
        self.n_exog = n_exog
        self.c_out = c_out
        self.dropout = dropout
        
        # Calculate number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        # Patch embedding for endogenous variable
        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            padding=0,
            dropout=dropout
        )
        
        # Exogenous embedding (if applicable)
        self.exog_embedding = None
        
        # Encoder layers
        self.encoder_layers = [
            TimeXerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ]
        
        # Output projection
        self.prediction_head = None
        
        # Global token for prediction
        self.global_token = None
    
    def set_weights(
        self,
        patch_embedding: Optional[Dict[str, np.ndarray]] = None,
        exog_embedding: Optional[Dict[str, np.ndarray]] = None,
        prediction_head: Optional[Dict[str, np.ndarray]] = None,
        global_token: Optional[np.ndarray] = None
    ):
        """
        设置模型权重
        Set model weights
        """
        if patch_embedding is not None:
            weight = patch_embedding.get('weight')
            bias = patch_embedding.get('bias')
            if weight is not None:
                self.patch_embedding.set_weights(weight, bias)
        
        self.exog_embedding = exog_embedding
        self.prediction_head = prediction_head
        self.global_token = global_token
    
    def _embed_exog(self, x: np.ndarray) -> np.ndarray:
        """应用外生变量嵌入"""
        if self.exog_embedding is None:
            # Simple projection
            batch, seq_len, n_features = x.shape
            if n_features < self.d_model:
                output = np.zeros((batch, seq_len, self.d_model))
                output[:, :, :n_features] = x
                return output
            else:
                return x[:, :, :self.d_model]
        
        weight = self.exog_embedding.get('weight')
        bias = self.exog_embedding.get('bias')
        
        if weight is not None:
            output = x @ weight.T
            if bias is not None:
                output = output + bias
            return output
        return x
    
    def _predict(self, x: np.ndarray) -> np.ndarray:
        """应用预测头"""
        batch, n_patches, d_model = x.shape
        
        if self.prediction_head is not None:
            weight = self.prediction_head.get('weight')
            bias = self.prediction_head.get('bias')
            
            # Flatten patches and project
            x_flat = x.reshape(batch, -1)
            
            if weight is not None:
                output = x_flat @ weight.T
                if bias is not None:
                    output = output + bias
            else:
                output = x_flat
            
            return output.reshape(batch, self.pred_len, self.c_out)
        else:
            # Simple projection using mean of patches
            mean_patch = np.mean(x, axis=1, keepdims=True)
            output = np.repeat(mean_patch, self.pred_len, axis=1)
            return output[:, :, :self.c_out]
    
    def forward(
        self,
        x_endog: np.ndarray,
        x_exog: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x_endog : np.ndarray
            内生变量，shape为 (batch, seq_len, enc_in)
        x_exog : np.ndarray, optional
            外生变量，shape为 (batch, seq_len + pred_len, n_exog)
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果，shape为 (batch, pred_len, c_out)
        """
        batch_size = x_endog.shape[0]
        
        # Patch embedding for endogenous variable
        patches = self.patch_embedding(x_endog)
        
        # Add global token if available
        if self.global_token is not None:
            global_token = np.broadcast_to(
                self.global_token[np.newaxis, np.newaxis, :],
                (batch_size, 1, self.d_model)
            )
            patches = np.concatenate([global_token, patches], axis=1)
        
        # Embed exogenous variables
        exog_embed = None
        if x_exog is not None and self.n_exog > 0:
            exog_embed = self._embed_exog(x_exog)
        
        # Pass through encoder layers
        enc_out = patches
        for enc_layer in self.encoder_layers:
            enc_out = enc_layer(enc_out, exog_embed)
        
        # Generate predictions
        predictions = self._predict(enc_out)
        
        return predictions
    
    def __call__(
        self,
        x_endog: np.ndarray,
        x_exog: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make the model callable"""
        return self.forward(x_endog, x_exog)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        Get model information
        """
        return {
            'model_type': 'TimeXer',
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'patch_len': self.patch_len,
            'stride': self.stride,
            'n_patches': self.n_patches,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'e_layers': self.e_layers,
            'd_ff': self.d_ff,
            'enc_in': self.enc_in,
            'n_exog': self.n_exog,
            'c_out': self.c_out,
            'has_patch_embedding': self.patch_embedding.weight is not None,
            'has_exog_embedding': self.exog_embedding is not None,
            'has_prediction_head': self.prediction_head is not None
        }
