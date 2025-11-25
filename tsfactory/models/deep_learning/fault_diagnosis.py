"""
Deep Learning Fault Diagnosis Models

深度学习故障诊断模型，用于样本分类
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from ..base_models import DLinearBaseModel, TransformerBaseModel, NLinearBaseModel


class FaultDiagnosisClassifier:
    """
    故障诊断分类器
    
    Sample-level classifier for fault diagnosis using time series data.
    Classifies entire sequences into different fault categories.
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int = 1,
        n_classes: int = 2,
        d_model: int = 64,
        model_type: str = 'dlinear'
    ):
        """
        Parameters:
        -----------
        seq_len : int
            输入序列长度 / Input sequence length
        n_features : int, default=1
            输入特征数 / Number of input features
        n_classes : int, default=2
            分类类别数 / Number of classes
        d_model : int, default=64
            模型维度 / Model dimension
        model_type : str, default='dlinear'
            基础模型类型 ('dlinear', 'nlinear', 'transformer')
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_model = d_model
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
        
        # Classification head
        self.classifier = None
        
        # Class names (optional)
        self.class_names = None
    
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
        
        # Load classifier
        self.classifier = state_dict.get('classifier')
        
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
        
        # Flatten features
        features_flat = features.reshape(batch_size, -1)
        
        # Apply classifier
        if self.classifier is not None:
            weight = self.classifier.get('weight')
            bias = self.classifier.get('bias')
            
            if weight is not None:
                # Adjust weight dimensions if needed
                if weight.shape[1] != features_flat.shape[1]:
                    # Truncate or pad features
                    if features_flat.shape[1] > weight.shape[1]:
                        features_flat = features_flat[:, :weight.shape[1]]
                    else:
                        padded = np.zeros((batch_size, weight.shape[1]))
                        padded[:, :features_flat.shape[1]] = features_flat
                        features_flat = padded
                
                logits = features_flat @ weight.T
                if bias is not None:
                    logits = logits + bias
            else:
                logits = np.zeros((batch_size, self.n_classes))
        else:
            # Random logits as placeholder
            logits = np.random.randn(batch_size, self.n_classes)
        
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
            'class_names': self.class_names
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
            'fault_names': self.fault_names
        }
