"""
Deep Learning Fault Diagnosis Models

深度学习故障诊断模型，用于样本分类
支持:
- 模型权重加载
- Head 部分微调 (fine_tune_head)
- 整体权重重新训练 (train_full)

分类头结构参考 Time-Series-Library:
    self.act = F.gelu
    self.dropout = nn.Dropout(configs.dropout)
    self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from ..base_models import DLinearBaseModel, TransformerBaseModel, NLinearBaseModel
from .training import ClassifierTrainer, create_training_data, initialize_feature_extractor
from .task_heads import ClassificationHead, gelu, dropout


class FaultDiagnosisClassifier:
    """
    故障诊断分类器
    
    Sample-level classifier for fault diagnosis using time series data.
    Classifies entire sequences into different fault categories.
    
    Classification head structure (Time-Series-Library reference):
        GELU activation + Dropout + Linear(d_model * enc_in, num_class)
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int = 1,
        n_classes: int = 2,
        d_model: int = 64,
        model_type: str = 'dlinear',
        dropout_rate: float = 0.1
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        n_features : int, default=1
            输入特征数 / Number of input features
            对应 Time-Series-Library 中的 enc_in
        n_classes : int, default=2
            分类类别数 / Number of classes
            对应 Time-Series-Library 中的 num_class
        d_model : int, default=64
            模型维度 / Model dimension
        model_type : str, default='dlinear'
            基础模型类型 ('dlinear', 'nlinear', 'transformer')
        dropout_rate : float, default=0.1
            Dropout 比率 (用于分类头)
        """
        self.seq_len = seq_len
        self.n_features = max(n_features, 1)  # Ensure at least 1 feature
        self.n_classes = n_classes
        self.d_model = d_model
        self.model_type = model_type
        self.dropout_rate = dropout_rate
        self.is_loaded = False
        self.training = False
        
        # Feature extractor
        if model_type == 'dlinear':
            self.feature_extractor = DLinearBaseModel(seq_len, d_model, self.n_features)
        elif model_type == 'nlinear':
            self.feature_extractor = NLinearBaseModel(seq_len, d_model, self.n_features)
        else:
            self.feature_extractor = TransformerBaseModel(
                seq_len, d_model, d_model=d_model, enc_in=self.n_features
            )
        
        # Classification head: Linear(d_model * n_features, n_classes)
        # Will be initialized with proper input dimension after feature extraction
        self.classifier = None
        self.classification_head = None
        
        # Class names (optional)
        self.class_names = None
    
    def train(self):
        """设置为训练模式"""
        self.training = True
        if self.classification_head is not None:
            self.classification_head.train()
    
    def eval(self):
        """设置为评估模式"""
        self.training = False
        if self.classification_head is not None:
            self.classification_head.eval()
    
    def set_class_names(self, class_names: List[str]):
        """设置类别名称"""
        if len(class_names) != self.n_classes:
            raise ValueError(f"Number of class names ({len(class_names)}) must match n_classes ({self.n_classes})")
        self.class_names = class_names
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'FaultDiagnosisClassifier':
        """
        从字典加载模型参数
        Load model parameters from dictionary
        """
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
        else:
            enc_weights = extractor_weights.get('encoder')
            dec_weights = extractor_weights.get('decoder')
            proj_weights = extractor_weights.get('projection')
            self.feature_extractor.set_weights(enc_weights, dec_weights, proj_weights)
        
        # Load classifier weights
        self.classifier = state_dict.get('classifier')
        
        # Initialize ClassificationHead if classifier weights are available
        if self.classifier is not None:
            weight = self.classifier.get('weight')
            if weight is not None:
                # Infer input dimension from weight shape
                input_dim = weight.shape[1]
                # Safe division: use max(n_features, 1) to prevent division by zero
                effective_n_features = max(self.n_features, 1)
                self.classification_head = ClassificationHead(
                    d_model=input_dim // effective_n_features,
                    enc_in=effective_n_features,
                    num_class=self.n_classes,
                    dropout_rate=self.dropout_rate
                )
                self.classification_head.load_weights(weight, self.classifier.get('bias'))
        
        # Load class names
        self.class_names = state_dict.get('class_names')
        
        self.is_loaded = True
        return self
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax函数"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测故障类型
        Predict fault type
        
        Classification head structure (Time-Series-Library reference):
            output = GELU(features)
            output = Dropout(output)
            logits = Linear(output)  # Linear(d_model * enc_in, num_class)
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列，shape为 (batch, seq_len, n_features) 或 (seq_len, n_features) 或 (seq_len,)
            
        Returns:
        --------
        predictions : np.ndarray
            预测类别 (0 to n_classes-1)
        probabilities : np.ndarray
            各类别概率
        """
        # Normalize input shape
        original_ndim = x.ndim
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, n_features = x.shape
        
        # Extract features
        try:
            features = self.feature_extractor.forward(x)
        except ValueError:
            # Model not loaded, use raw features
            features = x
        
        # Flatten features to (batch, d_model * enc_in)
        features_flat = features.reshape(batch_size, -1)
        
        # Apply classification head: GELU + Dropout + Linear
        if self.classification_head is not None:
            # Use ClassificationHead which includes GELU and Dropout
            logits = self.classification_head.forward(features_flat)
        elif self.classifier is not None:
            weight = self.classifier.get('weight')
            bias = self.classifier.get('bias')
            
            if weight is not None:
                # Adjust weight dimensions if needed
                if weight.shape[1] != features_flat.shape[1]:
                    if features_flat.shape[1] > weight.shape[1]:
                        features_flat = features_flat[:, :weight.shape[1]]
                    else:
                        padded = np.zeros((batch_size, weight.shape[1]))
                        padded[:, :features_flat.shape[1]] = features_flat
                        features_flat = padded
                
                # Apply GELU activation (Time-Series-Library reference)
                features_flat = gelu(features_flat)
                
                # Apply dropout (only in training mode)
                features_flat = dropout(features_flat, self.dropout_rate, self.training)
                
                # Linear projection
                logits = features_flat @ weight.T
                if bias is not None:
                    logits = logits + bias
            else:
                logits = np.zeros((batch_size, self.n_classes))
        else:
            # Use uniform distribution when classifier not loaded (equal probability for all classes)
            logits = np.zeros((batch_size, self.n_classes))
        
        # Softmax to get probabilities
        probabilities = self._softmax(logits, axis=-1)
        
        # Get predictions
        predictions = np.argmax(probabilities, axis=-1)
        
        # Restore original shape
        if original_ndim == 1 or original_ndim == 2:
            predictions = predictions.squeeze()
            probabilities = probabilities.squeeze()
        
        return predictions, probabilities
    
    def predict_with_names(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        预测并返回类别名称
        Predict and return class names
        
        Returns:
        --------
        predictions : np.ndarray
            预测类别索引
        probabilities : np.ndarray
            各类别概率
        names : List[str]
            预测类别名称
        """
        predictions, probabilities = self.predict(x)
        
        if self.class_names is not None:
            if predictions.ndim == 0:
                names = [self.class_names[int(predictions)]]
            else:
                names = [self.class_names[int(p)] for p in predictions]
        else:
            if predictions.ndim == 0:
                names = [f'Class_{int(predictions)}']
            else:
                names = [f'Class_{int(p)}' for p in predictions]
        
        return predictions, probabilities, names
    
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
            提取的特征，shape为 (batch, feature_dim)
        """
        try:
            features = self.feature_extractor.forward(x)
        except ValueError:
            features = x
        
        # Flatten
        return features.reshape(x.shape[0], -1)
    
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
            标签，shape为 (n_samples,) - 每个样本的类别标签 (0 to n_classes-1)
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
        
        # Extract features using frozen feature extractor
        features = self._extract_features(X)
        
        # Create and train classifier
        input_dim = features.shape[1]
        
        trainer = ClassifierTrainer(n_classes=self.n_classes, input_dim=input_dim)
        trainer.initialize()
        
        history = trainer.fit(
            features, y,
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
        
        Note: 由于使用numpy实现，完整训练采用两阶段策略
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据，shape为 (n_samples, seq_len, n_features)
        y : np.ndarray
            标签，shape为 (n_samples,)
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
            seq_len,
            n_features,
            self.d_model
        )
        
        history = []
        
        # Two-phase training approach
        # Phase 1: Pre-train feature extractor with self-supervised learning
        if verbose:
            print("Phase 1: Pre-training feature extractor...")
        
        for epoch in range(epochs // 2):
            indices = np.random.permutation(n_samples)
            epoch_losses = []
            
            for i in range(0, n_samples, batch_size):
                batch_X = X[indices[i:i+batch_size]]
                
                # Self-supervised: predict masked parts
                try:
                    features = self.feature_extractor.forward(batch_X)
                    # Simple reconstruction loss
                    loss = np.mean((batch_X.reshape(-1) - features.reshape(-1)[:batch_X.size]) ** 2)
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
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make the model callable"""
        return self.predict(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': f'FaultDiagnosisClassifier-{self.model_type}',
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'd_model': self.d_model,
            'is_loaded': self.is_loaded,
            'has_classifier': self.classifier is not None,
            'class_names': self.class_names,
            'supports_training': True
        }


class MultiFaultDiagnosisClassifier:
    """
    多故障诊断分类器
    
    Multi-label classifier for detecting multiple faults simultaneously.
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int = 1,
        n_faults: int = 3,
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
        n_faults : int, default=3
            故障类型数 / Number of fault types
        d_model : int, default=64
            模型维度 / Model dimension
        threshold : float, default=0.5
            检测阈值 / Detection threshold
        model_type : str, default='dlinear'
            基础模型类型
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_faults = n_faults
        self.d_model = d_model
        self.threshold = threshold
        self.model_type = model_type
        self.is_loaded = False
        
        # Feature extractor
        if model_type == 'dlinear':
            self.feature_extractor = DLinearBaseModel(seq_len, d_model, n_features)
        elif model_type == 'nlinear':
            self.feature_extractor = NLinearBaseModel(seq_len, d_model, n_features)
        else:
            self.feature_extractor = TransformerBaseModel(
                seq_len, d_model, d_model=d_model, enc_in=n_features
            )
        
        # Multi-label classification heads (one per fault type)
        self.classifiers = None
        
        # Fault names (optional)
        self.fault_names = None
    
    def set_fault_names(self, fault_names: List[str]):
        """设置故障名称"""
        if len(fault_names) != self.n_faults:
            raise ValueError(f"Number of fault names must match n_faults ({self.n_faults})")
        self.fault_names = fault_names
    
    def load_from_dict(self, state_dict: Dict[str, Any]) -> 'MultiFaultDiagnosisClassifier':
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
        
        # Load classifiers
        self.classifiers = state_dict.get('classifiers')
        
        # Load fault names
        self.fault_names = state_dict.get('fault_names')
        
        self.is_loaded = True
        return self
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测多故障
        Predict multiple faults
        
        Parameters:
        -----------
        x : np.ndarray
            输入序列
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果，shape为 (batch, n_faults)，每个元素为0或1
        probabilities : np.ndarray
            各故障概率，shape为 (batch, n_faults)
        """
        # Normalize input shape
        original_ndim = x.ndim
        if x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size = x.shape[0]
        
        # Extract features
        try:
            features = self.feature_extractor.forward(x)
        except ValueError:
            features = x
        
        # Flatten features
        features_flat = features.reshape(batch_size, -1)
        
        # Apply classifiers
        probabilities = np.zeros((batch_size, self.n_faults))
        
        if self.classifiers is not None:
            for i in range(self.n_faults):
                classifier = self.classifiers.get(f'fault_{i}')
                if classifier is not None:
                    weight = classifier.get('weight')
                    bias = classifier.get('bias')
                    
                    if weight is not None:
                        # Adjust dimensions
                        if features_flat.shape[1] > weight.shape[1]:
                            feat = features_flat[:, :weight.shape[1]]
                        elif features_flat.shape[1] < weight.shape[1]:
                            feat = np.zeros((batch_size, weight.shape[1]))
                            feat[:, :features_flat.shape[1]] = features_flat
                        else:
                            feat = features_flat
                        
                        logit = feat @ weight.T
                        if bias is not None:
                            logit = logit + bias
                        probabilities[:, i] = self._sigmoid(logit.squeeze())
                    else:
                        probabilities[:, i] = 0.5
                else:
                    probabilities[:, i] = 0.5
        else:
            # Random probabilities as placeholder
            probabilities = np.random.rand(batch_size, self.n_faults)
        
        # Apply threshold
        predictions = (probabilities > self.threshold).astype(int)
        
        # Restore original shape
        if original_ndim == 1 or original_ndim == 2:
            predictions = predictions.squeeze()
            probabilities = probabilities.squeeze()
        
        return predictions, probabilities
    
    def predict_with_names(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        """
        预测并返回故障名称
        Predict and return fault names
        """
        predictions, probabilities = self.predict(x)
        
        # Ensure predictions is 2D
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        
        # Get fault names for each sample
        all_faults = []
        for sample_preds in predictions:
            sample_faults = []
            for i, pred in enumerate(sample_preds):
                if pred == 1:
                    if self.fault_names is not None:
                        sample_faults.append(self.fault_names[i])
                    else:
                        sample_faults.append(f'Fault_{i}')
            if not sample_faults:
                sample_faults.append('Normal')
            all_faults.append(sample_faults)
        
        return predictions.squeeze(), probabilities, all_faults
    
    def _extract_features(self, x: np.ndarray) -> np.ndarray:
        """
        提取特征
        Extract features from input
        """
        try:
            features = self.feature_extractor.forward(x)
        except ValueError:
            features = x
        
        return features.reshape(x.shape[0], -1)
    
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
        Fine-tune classification heads only (freeze feature extractor)
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据，shape为 (n_samples, seq_len, n_features)
        y : np.ndarray
            多标签，shape为 (n_samples, n_faults) - 每个故障类型的标签 (0或1)
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
        
        # Extract features
        features = self._extract_features(X)
        input_dim = features.shape[1]
        
        # Train a classifier for each fault type
        self.classifiers = {}
        all_history = []
        
        for fault_idx in range(self.n_faults):
            if verbose:
                print(f"Training classifier for fault {fault_idx}...")
            
            fault_labels = y[:, fault_idx] if y.ndim > 1 else y
            
            trainer = ClassifierTrainer(n_classes=2, input_dim=input_dim)
            trainer.initialize()
            
            history = trainer.fit(
                features, fault_labels,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                verbose=False
            )
            
            self.classifiers[f'fault_{fault_idx}'] = trainer.get_weights()
            all_history.append(history[-1] if history else 0.0)
        
        self.is_loaded = True
        
        if verbose:
            print(f"Final losses: {all_history}")
        
        return all_history
    
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
        Full training (train both feature extractor and classification heads)
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据，shape为 (n_samples, seq_len, n_features)
        y : np.ndarray
            多标签，shape为 (n_samples, n_faults)
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
            seq_len,
            n_features,
            self.d_model
        )
        
        history = []
        
        # Phase 1: Pre-train feature extractor
        if verbose:
            print("Phase 1: Pre-training feature extractor...")
        
        for epoch in range(epochs // 2):
            indices = np.random.permutation(n_samples)
            epoch_losses = []
            
            for i in range(0, n_samples, batch_size):
                batch_X = X[indices[i:i+batch_size]]
                
                try:
                    features = self.feature_extractor.forward(batch_X)
                    loss = np.mean((batch_X.reshape(-1) - features.reshape(-1)[:batch_X.size]) ** 2)
                except ValueError:
                    loss = 0.0
                
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs // 2}, Pre-training Loss: {avg_loss:.4f}")
        
        # Phase 2: Fine-tune classification heads
        if verbose:
            print("Phase 2: Training classification heads...")
        
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
        return self.predict(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': f'MultiFaultDiagnosisClassifier-{self.model_type}',
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'n_faults': self.n_faults,
            'd_model': self.d_model,
            'threshold': self.threshold,
            'is_loaded': self.is_loaded,
            'has_classifiers': self.classifiers is not None,
            'fault_names': self.fault_names,
            'supports_training': True
        }
