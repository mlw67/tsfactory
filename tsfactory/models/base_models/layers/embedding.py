"""
Embedding Layers

嵌入层，用于时序模型的输入表示
"""

import numpy as np
from typing import Optional


class TokenEmbedding:
    """
    Token嵌入层
    
    Converts input features to higher dimensional embeddings using 1D convolution-like operation.
    """
    
    def __init__(self, c_in: int, d_model: int, kernel_size: int = 3, padding: int = 1):
        """
        Parameters:
        -----------
        c_in : int
            输入特征维度 / Input feature dimension
        d_model : int
            嵌入维度 / Embedding dimension
        kernel_size : int, default=3
            卷积核大小 / Kernel size
        padding : int, default=1
            填充大小 / Padding size
        """
        self.c_in = c_in
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Weights (will be set when loading)
        self.weight = None
        self.bias = None
    
    def set_weights(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """设置权重"""
        self.weight = weight
        self.bias = bias
    
    def _conv1d(self, x: np.ndarray) -> np.ndarray:
        """
        一维卷积操作
        1D convolution operation
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch, seq_len, c_in)
            
        Returns:
        --------
        output : np.ndarray
            输出，shape为 (batch, seq_len, d_model)
        """
        if self.weight is None:
            # Return identity-like transformation
            batch, seq_len, c_in = x.shape
            output = np.zeros((batch, seq_len, self.d_model))
            min_dim = min(c_in, self.d_model)
            output[:, :, :min_dim] = x[:, :, :min_dim]
            return output
        
        batch, seq_len, c_in = x.shape
        
        # Pad input
        x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)), mode='constant')
        
        # Simple linear projection as fallback (conv1d-like)
        output = np.zeros((batch, seq_len, self.d_model))
        for i in range(seq_len):
            window = x_padded[:, i:i+self.kernel_size, :]
            window_flat = window.reshape(batch, -1)
            # Use weight to project
            if window_flat.shape[1] <= self.weight.shape[1]:
                output[:, i, :] = window_flat @ self.weight[:, :window_flat.shape[1]].T
            else:
                output[:, i, :] = window_flat[:, :self.weight.shape[1]] @ self.weight.T
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch, seq_len, c_in)
            
        Returns:
        --------
        output : np.ndarray
            输出，shape为 (batch, seq_len, d_model)
        """
        return self._conv1d(x)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)


class PositionalEmbedding:
    """
    位置嵌入层
    
    Fixed positional embeddings using sinusoidal functions.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Parameters:
        -----------
        d_model : int
            嵌入维度 / Embedding dimension
        max_len : int, default=5000
            最大序列长度 / Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len
        
        # Pre-compute positional encodings
        self.pe = self._generate_pe(max_len, d_model)
    
    def _generate_pe(self, length: int, d_model: int) -> np.ndarray:
        """生成位置编码"""
        pe = np.zeros((length, d_model))
        position = np.arange(0, length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        n_cos_positions = (d_model - 1) // 2 + (d_model % 2 == 0)
        pe[:, 1::2] = np.cos(position * div_term[:n_cos_positions])
        
        return pe
    
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
        batch, seq_len, d_model = x.shape
        
        if seq_len > self.max_len:
            pe = self._generate_pe(seq_len, d_model)
        else:
            pe = self.pe[:seq_len, :]
        
        return pe[np.newaxis, :, :]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)


class TemporalEmbedding:
    """
    时间嵌入层
    
    Embeds temporal features like minute, hour, weekday, day, month.
    """
    
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        """
        Parameters:
        -----------
        d_model : int
            嵌入维度 / Embedding dimension
        embed_type : str, default='fixed'
            嵌入类型 ('fixed', 'learned') / Embedding type
        freq : str, default='h'
            时间频率 ('h', 't', 's', 'm', 'a', 'w', 'd', 'b')
            Frequency: hourly, minutely, secondly, monthly, annually, weekly, daily, business days
        """
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        
        # Embedding tables
        self.minute_embed = None
        self.hour_embed = None
        self.weekday_embed = None
        self.day_embed = None
        self.month_embed = None
    
    def set_embeddings(
        self,
        minute_embed: Optional[np.ndarray] = None,
        hour_embed: Optional[np.ndarray] = None,
        weekday_embed: Optional[np.ndarray] = None,
        day_embed: Optional[np.ndarray] = None,
        month_embed: Optional[np.ndarray] = None
    ):
        """设置嵌入表"""
        self.minute_embed = minute_embed
        self.hour_embed = hour_embed
        self.weekday_embed = weekday_embed
        self.day_embed = day_embed
        self.month_embed = month_embed
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            时间特征，shape为 (batch, seq_len, n_time_features)
            时间特征顺序: [minute, hour, weekday, day, month]
            
        Returns:
        --------
        output : np.ndarray
            时间嵌入，shape为 (batch, seq_len, d_model)
        """
        batch, seq_len, n_features = x.shape
        output = np.zeros((batch, seq_len, self.d_model))
        
        # Add embeddings based on available features and embedding tables
        feature_idx = 0
        
        if n_features > feature_idx and self.minute_embed is not None:
            minute_idx = x[:, :, feature_idx].astype(int) % self.minute_embed.shape[0]
            for b in range(batch):
                for s in range(seq_len):
                    output[b, s, :] += self.minute_embed[minute_idx[b, s]]
            feature_idx += 1
        
        if n_features > feature_idx and self.hour_embed is not None:
            hour_idx = x[:, :, feature_idx].astype(int) % self.hour_embed.shape[0]
            for b in range(batch):
                for s in range(seq_len):
                    output[b, s, :] += self.hour_embed[hour_idx[b, s]]
            feature_idx += 1
        
        if n_features > feature_idx and self.weekday_embed is not None:
            weekday_idx = x[:, :, feature_idx].astype(int) % self.weekday_embed.shape[0]
            for b in range(batch):
                for s in range(seq_len):
                    output[b, s, :] += self.weekday_embed[weekday_idx[b, s]]
            feature_idx += 1
        
        if n_features > feature_idx and self.day_embed is not None:
            day_idx = x[:, :, feature_idx].astype(int) % self.day_embed.shape[0]
            for b in range(batch):
                for s in range(seq_len):
                    output[b, s, :] += self.day_embed[day_idx[b, s]]
            feature_idx += 1
        
        if n_features > feature_idx and self.month_embed is not None:
            month_idx = x[:, :, feature_idx].astype(int) % self.month_embed.shape[0]
            for b in range(batch):
                for s in range(seq_len):
                    output[b, s, :] += self.month_embed[month_idx[b, s]]
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)


class DataEmbedding:
    """
    数据嵌入层
    
    Combines value embedding, positional embedding, and temporal embedding.
    """
    
    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        c_in : int
            输入特征维度 / Input feature dimension
        d_model : int
            嵌入维度 / Embedding dimension
        embed_type : str, default='fixed'
            嵌入类型 / Embedding type
        freq : str, default='h'
            时间频率 / Time frequency
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.c_in = c_in
        self.d_model = d_model
        self.dropout = dropout
        
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model, embed_type, freq)
    
    def forward(
        self,
        x: np.ndarray,
        x_mark: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入值，shape为 (batch, seq_len, c_in)
        x_mark : np.ndarray, optional
            时间标记，shape为 (batch, seq_len, n_time_features)
            
        Returns:
        --------
        output : np.ndarray
            嵌入输出，shape为 (batch, seq_len, d_model)
        """
        output = self.value_embedding(x) + self.position_embedding(x)
        
        if x_mark is not None:
            output = output + self.temporal_embedding(x_mark)
        
        return output
    
    def __call__(
        self,
        x: np.ndarray,
        x_mark: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x, x_mark)


class PatchEmbedding:
    """
    Patch嵌入层
    
    Divides input sequence into patches and embeds them.
    Used in models like TimeXer and PatchTST.
    """
    
    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        padding: int = 0,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        d_model : int
            嵌入维度 / Embedding dimension
        patch_len : int
            Patch长度 / Patch length
        stride : int
            步长 / Stride
        padding : int, default=0
            填充 / Padding
        dropout : float, default=0.1
            Dropout率 / Dropout rate
        """
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        
        # Projection weight
        self.weight = None
        self.bias = None
    
    def set_weights(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        """设置权重"""
        self.weight = weight
        self.bias = bias
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward pass
        
        Parameters:
        -----------
        x : np.ndarray
            输入，shape为 (batch, seq_len, n_features)
            
        Returns:
        --------
        output : np.ndarray
            Patch嵌入，shape为 (batch, n_patches, d_model)
        """
        batch, seq_len, n_features = x.shape
        
        # Pad if necessary
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, self.padding), (0, 0)), mode='constant')
            seq_len = x.shape[1]
        
        # Calculate number of patches
        n_patches = (seq_len - self.patch_len) // self.stride + 1
        
        # Extract patches
        patches = np.zeros((batch, n_patches, self.patch_len * n_features))
        for i in range(n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches[:, i, :] = x[:, start:end, :].reshape(batch, -1)
        
        # Project to d_model
        if self.weight is not None:
            output = patches @ self.weight.T
            if self.bias is not None:
                output = output + self.bias
        else:
            # Default projection
            output = np.zeros((batch, n_patches, self.d_model))
            min_dim = min(patches.shape[2], self.d_model)
            output[:, :, :min_dim] = patches[:, :, :min_dim]
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the layer callable"""
        return self.forward(x)
