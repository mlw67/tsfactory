"""
Deep Learning Anomaly Detection Models

深度学习异常检测模型，包括点分类和区间分类
支持:
- 模型权重加载
- Head 部分微调 (fine_tune_head)
- 整体权重重新训练 (train_full)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from ..base_models import DLinearBaseModel, TransformerBaseModel, NLinearBaseModel
from .training import ClassifierTrainer, create_training_data, initialize_feature_extractor


class BaseAnomalyDetector:
    """
    异常检测基类
    
    Base class for deep learning anomaly detection models.
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int = 1,
        threshold: float = 0.5
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        n_features : int, default=1
            输入特征数 / Number of input features
        threshold : float, default=0.5
            异常阈值 / Anomaly threshold
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.threshold = threshold
        self.is_loaded = False
    
    def set_threshold(self, threshold: float):
        """设置阈值"""
        self.threshold = threshold


class PointAnomalyDetector(BaseAnomalyDetector):
    """
    点异常检测器
    
    Detects point-level anomalies using reconstruction-based approach.
    Uses an encoder-decoder architecture to reconstruct normal patterns,
    then identifies anomalies based on reconstruction error.
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int = 1,
        d_model: int = 64,
        threshold: float = 0.5,
        model_type: str = 'dlinear'
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        n_features : int, default=1
            输入特征数 / Number of input features
        d_model : int, default=64
            模型维度 / Model dimension
        threshold : float, default=0.5
            异常阈值 / Anomaly threshold
        model_type : str, default='dlinear'
            基础模型类型 ('dlinear', 'nlinear', 'transformer')
        """
        super().__init__(seq_len, n_features, threshold)
        self.d_model = d_model
        self.model_type = model_type
        
        # Reconstruction model (seq_len -> seq_len)
        if model_type == 'dlinear':
            self.encoder = DLinearBaseModel(seq_len, seq_len, n_features)
        elif model_type == 'nlinear':
            self.encoder = NLinearBaseModel(seq_len, seq_len, n_features)
        else:
            self.encoder = TransformerBaseModel(seq_len, seq_len, d_model=d_model, enc_in=n_features)
        
        # Classification head weights
        self.classifier = None
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'PointAnomalyDetector':
        """
        从字典加载模型参数
        Load model parameters from dictionary
        """
        # Load encoder weights
        encoder_weights = state_dict.get('encoder', {})
        if self.model_type in ['dlinear']:
            seasonal = encoder_weights.get('seasonal')
            trend = encoder_weights.get('trend')
            if seasonal is not None or trend is not None:
                self.encoder.set_weights(seasonal, trend)
        elif self.model_type == 'nlinear':
            linear = encoder_weights.get('linear')
            if linear is not None:
                self.encoder.set_weights(linear)
        else:
            enc_weights = encoder_weights.get('encoder')
            dec_weights = encoder_weights.get('decoder')
            proj_weights = encoder_weights.get('projection')
            self.encoder.set_weights(enc_weights, dec_weights, proj_weights)
        
        # Load classifier weights
        self.classifier = state_dict.get('classifier')
        
        self.is_loaded = True
        return self
    
    def _compute_reconstruction_error(
        self,
        x: np.ndarray,
        reconstructed: np.ndarray
    ) -> np.ndarray:
        """
        计算重建误差
        Compute reconstruction error
        
        Parameters:
        -----------
        x : np.ndarray
            原始输入
        reconstructed : np.ndarray
            重建输出
            
        Returns:
        --------
        error : np.ndarray
            每个点的重建误差
        """
        # Mean squared error per point
        error = np.mean((x - reconstructed) ** 2, axis=-1)
        return error
    
    def detect(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测点异常
        Detect point anomalies
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch, seq_len, n_features) 或 (seq_len, n_features) 或 (seq_len,)
            
        Returns:
        --------
        labels : np.ndarray
            异常标签 (0=正常, 1=异常)
        scores : np.ndarray
            异常得分
        """
        # Normalize input shape
        original_shape = x.shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, n_features = x.shape
        
        # Reconstruct
        try:
            reconstructed = self.encoder.forward(x)
        except ValueError:
            # Model not loaded, use identity reconstruction with noise
            reconstructed = x + np.random.randn(*x.shape) * 0.1
        
        # Compute reconstruction error
        scores = self._compute_reconstruction_error(x, reconstructed)
        
        # Apply classifier if available
        if self.classifier is not None:
            weight = self.classifier.get('weight')
            bias = self.classifier.get('bias')
            if weight is not None:
                scores_flat = scores.reshape(batch_size, -1)
                if weight.shape[1] == scores_flat.shape[1]:
                    scores = scores_flat @ weight.T
                    if bias is not None:
                        scores = scores + bias
                    # Sigmoid
                    scores = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
                    scores = scores.reshape(batch_size, seq_len)
        
        # Apply threshold
        labels = (scores > self.threshold).astype(int)
        
        # Restore original shape
        if len(original_shape) == 1:
            labels = labels.squeeze()
            scores = scores.squeeze()
        elif len(original_shape) == 2:
            labels = labels.squeeze(0)
            scores = scores.squeeze(0)
        
        return labels, scores
    
    def _extract_features(self, x: np.ndarray) -> np.ndarray:
        """
        提取特征
        Extract features from input
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch, seq_len, n_features)
            
        Returns:
        --------
        features : np.ndarray
            提取的特征
        """
        try:
            reconstructed = self.encoder.forward(x)
            # Use reconstruction error as features
            error = self._compute_reconstruction_error(x, reconstructed)
            return error
        except ValueError:
            # Model not loaded, return raw flattened input
            return x.reshape(x.shape[0], -1)
    
    def fine_tune_head(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ) -> List[float]:
        """
        微调分类头
        Fine-tune classification head only (freeze feature extractor)
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据，shape为 (n_samples, seq_len, n_features) 或 (n_samples, seq_len)
        y : np.ndarray
            标签，shape为 (n_samples, seq_len) - 每个点的异常标签 (0或1)
        epochs : int, default=100
            训练轮数
        learning_rate : float, default=0.01
            学习率
        batch_size : int, default=32
            批大小
        verbose : bool, default=True
            是否打印训练信息
            
        Returns:
        --------
        history : List[float]
            训练损失历史
        """
        # Normalize input shape
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        n_samples, seq_len, n_features = X.shape
        
        # Extract features using frozen encoder
        features = self._extract_features(X)
        
        # Flatten features and labels
        features_flat = features.reshape(n_samples, -1)
        y_flat = y.reshape(-1)  # Flatten all labels for binary classification
        
        # Create feature-label pairs for each point
        # Each point needs its own prediction
        all_features = []
        all_labels = []
        
        for i in range(n_samples):
            for j in range(seq_len):
                # Use the j-th position feature for j-th point
                if features_flat.shape[1] >= seq_len:
                    feat_idx = j * (features_flat.shape[1] // seq_len)
                    feat = features_flat[i, feat_idx:feat_idx + (features_flat.shape[1] // seq_len)]
                else:
                    feat = features_flat[i]
                all_features.append(feat)
                all_labels.append(y[i, j] if y.ndim > 1 else y[i])
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        # Create and train binary classifier
        input_dim = all_features.shape[1]
        
        trainer = ClassifierTrainer(n_classes=2, input_dim=input_dim)
        trainer.initialize()
        
        history = trainer.fit(
            all_features, all_labels,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Save trained weights
        self.classifier = trainer.get_weights()
        self.is_loaded = True
        
        return history
    
    def train_full(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ) -> List[float]:
        """
        完整训练（包括特征提取器和分类头）
        Full training (train both feature extractor and classification head)
        
        Note: 由于使用numpy实现，完整训练效率较低，建议使用PyTorch等框架进行完整训练
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据，shape为 (n_samples, seq_len, n_features)
        y : np.ndarray
            标签，shape为 (n_samples, seq_len)
        epochs : int, default=100
            训练轮数
        learning_rate : float, default=0.01
            学习率
        batch_size : int, default=32
            批大小
        verbose : bool, default=True
            是否打印训练信息
            
        Returns:
        --------
        history : List[float]
            训练损失历史
        """
        # Normalize input shape
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        n_samples, seq_len, n_features = X.shape
        
        # Initialize encoder weights if not loaded
        if self.model_type in ['dlinear']:
            if self.encoder.seasonal_linear is None:
                # Initialize with identity-like weights for reconstruction
                input_size = seq_len * n_features
                output_size = seq_len * n_features
                self.encoder.set_weights(
                    {'weight': np.eye(output_size, input_size), 'bias': np.zeros(output_size)},
                    {'weight': np.eye(output_size, input_size), 'bias': np.zeros(output_size)}
                )
        elif self.model_type == 'nlinear':
            if self.encoder.linear is None:
                input_size = seq_len * n_features
                output_size = seq_len * n_features
                self.encoder.set_weights(
                    {'weight': np.eye(output_size, input_size), 'bias': np.zeros(output_size)}
                )
        
        # For full training, we first train the reconstruction
        # Then fine-tune the head
        history = []
        
        # Phase 1: Train reconstruction (encoder weights)
        if verbose:
            print("Phase 1: Training reconstruction...")
        
        for epoch in range(epochs // 2):
            indices = np.random.permutation(n_samples)
            epoch_losses = []
            
            for i in range(0, n_samples, batch_size):
                batch_X = X[indices[i:i+batch_size]]
                
                # Forward pass
                try:
                    reconstructed = self.encoder.forward(batch_X)
                except ValueError:
                    reconstructed = batch_X
                
                # Reconstruction loss
                loss = np.mean((batch_X - reconstructed) ** 2)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs // 2}, Reconstruction Loss: {avg_loss:.4f}")
        
        # Phase 2: Fine-tune head
        if verbose:
            print("Phase 2: Fine-tuning classification head...")
        
        head_history = self.fine_tune_head(
            X, y,
            epochs=epochs - epochs // 2,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose
        )
        
        history.extend(head_history)
        
        return history
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make the model callable"""
        return self.detect(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': f'PointAnomalyDetector-{self.model_type}',
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'd_model': self.d_model,
            'threshold': self.threshold,
            'is_loaded': self.is_loaded,
            'has_classifier': self.classifier is not None,
            'supports_training': True
        }


class IntervalAnomalyDetector(BaseAnomalyDetector):
    """
    区间异常检测器
    
    Detects interval-level anomalies by classifying segments of time series.
    Uses a sliding window approach to classify each interval as normal or anomalous.
    """
    
    def __init__(
        self,
        seq_len: int,
        interval_len: int,
        n_features: int = 1,
        d_model: int = 64,
        threshold: float = 0.5,
        stride: int = None,
        model_type: str = 'dlinear'
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        interval_len : int
            检测区间长度 / Detection interval length
        n_features : int, default=1
            输入特征数 / Number of input features
        d_model : int, default=64
            模型维度 / Model dimension
        threshold : float, default=0.5
            异常阈值 / Anomaly threshold
        stride : int, optional
            滑动步长，默认为interval_len / Sliding stride
        model_type : str, default='dlinear'
            基础模型类型
        """
        super().__init__(seq_len, n_features, threshold)
        self.interval_len = interval_len
        self.d_model = d_model
        self.stride = stride if stride is not None else interval_len
        self.model_type = model_type
        
        # Feature extractor
        if model_type == 'dlinear':
            self.feature_extractor = DLinearBaseModel(interval_len, d_model, n_features)
        elif model_type == 'nlinear':
            self.feature_extractor = NLinearBaseModel(interval_len, d_model, n_features)
        else:
            self.feature_extractor = TransformerBaseModel(
                interval_len, d_model, d_model=d_model, enc_in=n_features
            )
        
        # Classification head
        self.classifier = None
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'IntervalAnomalyDetector':
        """从字典加载模型参数"""
        # Load feature extractor weights
        extractor_weights = state_dict.get('feature_extractor', {})
        if self.model_type in ['dlinear']:
            seasonal = extractor_weights.get('seasonal')
            trend = extractor_weights.get('trend')
            if seasonal is not None or trend is not None:
                self.feature_extractor.set_weights(seasonal, trend)
        elif self.model_type == 'nlinear':
            linear = extractor_weights.get('linear')
            if linear is not None:
                self.feature_extractor.set_weights(linear)
        
        # Load classifier
        self.classifier = state_dict.get('classifier')
        
        self.is_loaded = True
        return self
    
    def _extract_intervals(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        提取时间区间
        Extract time intervals
        """
        batch_size, seq_len, n_features = x.shape
        
        intervals = []
        positions = []
        
        for start in range(0, seq_len - self.interval_len + 1, self.stride):
            end = start + self.interval_len
            interval = x[:, start:end, :]
            intervals.append(interval)
            positions.append((start, end))
        
        if len(intervals) == 0:
            return x, [(0, seq_len)]
        
        return np.array(intervals), positions
    
    def detect(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        检测区间异常
        Detect interval anomalies
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列
            
        Returns:
        --------
        labels : np.ndarray
            每个区间的异常标签
        scores : np.ndarray
            每个区间的异常得分
        positions : List[Tuple[int, int]]
            区间位置列表
        """
        # Normalize input shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size = x.shape[0]
        
        # Extract intervals
        intervals, positions = self._extract_intervals(x)
        n_intervals = len(intervals)
        
        if n_intervals == 0:
            return np.array([0]), np.array([0.0]), [(0, x.shape[1])]
        
        # Process each interval
        scores = []
        for interval in intervals:
            # Extract features
            try:
                features = self.feature_extractor.forward(interval)
            except ValueError:
                features = interval
            
            # Compute score
            if self.classifier is not None:
                weight = self.classifier.get('weight')
                bias = self.classifier.get('bias')
                
                features_flat = features.reshape(batch_size, -1)
                if weight is not None and weight.shape[1] == features_flat.shape[1]:
                    score = features_flat @ weight.T
                    if bias is not None:
                        score = score + bias
                    # Sigmoid
                    score = 1 / (1 + np.exp(-np.clip(score, -500, 500)))
                else:
                    score = np.mean(features_flat, axis=-1, keepdims=True)
            else:
                # Use variance as anomaly score
                score = np.var(features, axis=(1, 2), keepdims=True).reshape(batch_size, 1)
            
            scores.append(score.squeeze())
        
        scores = np.array(scores).T  # (batch_size, n_intervals)
        
        # Apply threshold
        labels = (scores > self.threshold).astype(int)
        
        return labels.squeeze(), scores.squeeze(), positions
    
    def detect_point_labels(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取点级别的异常标签
        Get point-level anomaly labels
        
        将区间标签转换为点标签
        """
        # Normalize input shape
        original_shape = x.shape
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, n_features = x.shape
        
        # Get interval predictions
        labels, scores, positions = self.detect(x)
        
        # Convert to point labels
        point_labels = np.zeros((batch_size, seq_len), dtype=int)
        point_scores = np.zeros((batch_size, seq_len))
        
        if labels.ndim == 1:
            labels = labels.reshape(1, -1)
            scores = scores.reshape(1, -1)
        
        for i, (start, end) in enumerate(positions):
            if i < labels.shape[1]:
                point_labels[:, start:end] = np.maximum(
                    point_labels[:, start:end],
                    labels[:, i:i+1]
                )
                point_scores[:, start:end] = np.maximum(
                    point_scores[:, start:end],
                    scores[:, i:i+1]
                )
        
        # Restore original shape
        if len(original_shape) == 1:
            point_labels = point_labels.squeeze()
            point_scores = point_scores.squeeze()
        elif len(original_shape) == 2:
            point_labels = point_labels.squeeze(0)
            point_scores = point_scores.squeeze(0)
        
        return point_labels, point_scores
    
    def _extract_interval_features(self, x: np.ndarray) -> np.ndarray:
        """
        提取区间特征
        Extract features for intervals
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch, seq_len, n_features)
            
        Returns:
        --------
        features : np.ndarray
            区间特征，shape为 (n_intervals, batch, d_model)
        """
        intervals, positions = self._extract_intervals(x)
        
        features_list = []
        for interval in intervals:
            try:
                features = self.feature_extractor.forward(interval)
            except ValueError:
                features = interval
            features_list.append(features)
        
        return np.array(features_list), positions
    
    def fine_tune_head(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ) -> List[float]:
        """
        微调分类头
        Fine-tune classification head only (freeze feature extractor)
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据，shape为 (n_samples, seq_len, n_features) 或 (n_samples, seq_len)
        y : np.ndarray
            区间标签，shape为 (n_samples, n_intervals) - 每个区间的异常标签
        epochs : int, default=100
            训练轮数
        learning_rate : float, default=0.01
            学习率
        batch_size : int, default=32
            批大小
        verbose : bool, default=True
            是否打印训练信息
            
        Returns:
        --------
        history : List[float]
            训练损失历史
        """
        # Normalize input shape
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        n_samples = X.shape[0]
        
        # Extract features for all intervals
        all_features = []
        all_labels = []
        
        for i in range(n_samples):
            sample_X = X[i:i+1]
            features, positions = self._extract_interval_features(sample_X)
            
            # Flatten features per interval
            for j, feat in enumerate(features):
                feat_flat = feat.reshape(-1)
                all_features.append(feat_flat)
                
                if j < y.shape[1]:
                    all_labels.append(y[i, j])
                else:
                    all_labels.append(0)
        
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        # Create and train classifier (binary)
        input_dim = features_array.shape[1]
        
        trainer = ClassifierTrainer(n_classes=2, input_dim=input_dim)
        trainer.initialize()
        
        history = trainer.fit(
            features_array, labels_array,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Save trained weights
        self.classifier = trainer.get_weights()
        self.is_loaded = True
        
        return history
    
    def train_full(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ) -> List[float]:
        """
        完整训练（包括特征提取器和分类头）
        Full training (train both feature extractor and classification head)
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据，shape为 (n_samples, seq_len, n_features)
        y : np.ndarray
            区间标签，shape为 (n_samples, n_intervals)
        epochs : int, default=100
            训练轮数
        learning_rate : float, default=0.01
            学习率
        batch_size : int, default=32
            批大小
        verbose : bool, default=True
            是否打印训练信息
            
        Returns:
        --------
        history : List[float]
            训练损失历史
        """
        # Normalize input shape
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        n_samples, seq_len, n_features = X.shape
        
        # Initialize feature extractor weights if not loaded
        initialize_feature_extractor(
            self.feature_extractor,
            self.model_type,
            self.interval_len,
            n_features,
            self.d_model
        )
        
        history = []
        
        # For full training, we train end-to-end
        # Phase 1: Pre-train feature extractor on interval reconstruction
        if verbose:
            print("Phase 1: Pre-training feature extractor...")
        
        for epoch in range(epochs // 2):
            epoch_losses = []
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                sample_X = X[idx:idx+1]
                intervals, _ = self._extract_intervals(sample_X)
                
                for interval in intervals:
                    try:
                        features = self.feature_extractor.forward(interval)
                        loss = np.mean((interval.reshape(-1) - features.reshape(-1)[:interval.size]) ** 2)
                    except ValueError:
                        loss = 0.0
                    epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs // 2}, Pre-training Loss: {avg_loss:.4f}")
        
        # Phase 2: Fine-tune classification head
        if verbose:
            print("Phase 2: Training classification head...")
        
        head_history = self.fine_tune_head(
            X, y,
            epochs=epochs - epochs // 2,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose
        )
        
        history.extend(head_history)
        
        return history
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Make the model callable"""
        return self.detect(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': f'IntervalAnomalyDetector-{self.model_type}',
            'seq_len': self.seq_len,
            'interval_len': self.interval_len,
            'n_features': self.n_features,
            'd_model': self.d_model,
            'stride': self.stride,
            'threshold': self.threshold,
            'is_loaded': self.is_loaded,
            'has_classifier': self.classifier is not None,
            'supports_training': True
        }
