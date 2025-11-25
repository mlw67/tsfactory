"""
Training Utilities for Deep Learning Models

深度学习模型训练工具，支持 head 部分微调和整体权重重新训练
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable


class TrainingMixin:
    """
    训练混入类
    
    Mixin class providing training capabilities for deep learning models.
    Supports head-only fine-tuning and full model training.
    """
    
    def _compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        loss_type: str = 'mse'
    ) -> float:
        """
        计算损失
        Compute loss
        
        Parameters:
        -----------
        predictions : np.ndarray
            预测值
        targets : np.ndarray
            目标值
        loss_type : str, default='mse'
            损失类型 ('mse', 'mae', 'cross_entropy', 'binary_cross_entropy')
            
        Returns:
        --------
        loss : float
            损失值
        """
        if loss_type == 'mse':
            return np.mean((predictions - targets) ** 2)
        elif loss_type == 'mae':
            return np.mean(np.abs(predictions - targets))
        elif loss_type == 'cross_entropy':
            # For multi-class classification: -sum(y * log(p)) averaged over samples
            eps = 1e-8
            predictions = np.clip(predictions, eps, 1 - eps)
            return -np.mean(np.sum(targets * np.log(predictions), axis=-1))
        elif loss_type == 'binary_cross_entropy':
            # For binary classification
            eps = 1e-8
            predictions = np.clip(predictions, eps, 1 - eps)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _compute_gradients_linear(
        self,
        x: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        loss_type: str = 'mse'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算线性层梯度
        Compute gradients for linear layer
        
        Parameters:
        -----------
        x : np.ndarray
            输入特征，shape为 (batch, input_dim)
        predictions : np.ndarray
            预测值，shape为 (batch, output_dim)
        targets : np.ndarray
            目标值，shape为 (batch, output_dim)
        loss_type : str, default='mse'
            损失类型
            
        Returns:
        --------
        grad_weight : np.ndarray
            权重梯度
        grad_bias : np.ndarray
            偏置梯度
        """
        batch_size = x.shape[0]
        
        if loss_type == 'mse':
            # dL/dy = 2 * (y - t) / n
            grad_output = 2 * (predictions - targets) / batch_size
        elif loss_type == 'mae':
            grad_output = np.sign(predictions - targets) / batch_size
        elif loss_type in ['cross_entropy', 'binary_cross_entropy']:
            # For softmax + cross entropy: dL/dy = y - t
            grad_output = (predictions - targets) / batch_size
        else:
            grad_output = 2 * (predictions - targets) / batch_size
        
        # dL/dW = dL/dy * x^T
        grad_weight = grad_output.T @ x
        
        # dL/db = sum(dL/dy)
        grad_bias = np.sum(grad_output, axis=0)
        
        return grad_weight, grad_bias
    
    def _update_weights(
        self,
        weights: Dict[str, np.ndarray],
        grad_weight: np.ndarray,
        grad_bias: np.ndarray,
        learning_rate: float
    ) -> Dict[str, np.ndarray]:
        """
        更新权重
        Update weights using gradients
        
        Parameters:
        -----------
        weights : dict
            当前权重
        grad_weight : np.ndarray
            权重梯度
        grad_bias : np.ndarray
            偏置梯度
        learning_rate : float
            学习率
            
        Returns:
        --------
        updated_weights : dict
            更新后的权重
        """
        updated_weights = {}
        
        if 'weight' in weights and weights['weight'] is not None:
            updated_weights['weight'] = weights['weight'] - learning_rate * grad_weight
        else:
            # Initialize weight with proper shape from gradient
            updated_weights['weight'] = np.zeros_like(grad_weight) - learning_rate * grad_weight
        
        if 'bias' in weights and weights['bias'] is not None:
            updated_weights['bias'] = weights['bias'] - learning_rate * grad_bias
        else:
            # Initialize bias with proper shape from gradient
            updated_weights['bias'] = np.zeros_like(grad_bias) - learning_rate * grad_bias
        
        return updated_weights
    
    def _initialize_weights(
        self,
        input_dim: int,
        output_dim: int,
        init_type: str = 'xavier'
    ) -> Dict[str, np.ndarray]:
        """
        初始化权重
        Initialize weights
        
        Parameters:
        -----------
        input_dim : int
            输入维度
        output_dim : int
            输出维度
        init_type : str, default='xavier'
            初始化类型 ('xavier', 'he', 'zeros', 'random')
            
        Returns:
        --------
        weights : dict
            初始化的权重
        """
        if init_type == 'xavier':
            scale = np.sqrt(2.0 / (input_dim + output_dim))
            weight = np.random.randn(output_dim, input_dim) * scale
        elif init_type == 'he':
            scale = np.sqrt(2.0 / input_dim)
            weight = np.random.randn(output_dim, input_dim) * scale
        elif init_type == 'zeros':
            weight = np.zeros((output_dim, input_dim))
        else:
            weight = np.random.randn(output_dim, input_dim) * 0.01
        
        bias = np.zeros(output_dim)
        
        return {'weight': weight, 'bias': bias}


def initialize_feature_extractor(
    feature_extractor,
    model_type: str,
    seq_len: int,
    n_features: int,
    output_dim: int
):
    """
    初始化特征提取器权重
    Initialize feature extractor weights
    
    Parameters:
    -----------
    feature_extractor : BaseModel
        特征提取器模型
    model_type : str
        模型类型 ('dlinear', 'nlinear', 'transformer')
    seq_len : int
        序列长度
    n_features : int
        特征数
    output_dim : int
        输出维度
    """
    input_size = seq_len * n_features
    output_size = output_dim * n_features
    
    if model_type in ['dlinear']:
        if feature_extractor.seasonal_linear is None:
            feature_extractor.set_weights(
                {'weight': np.random.randn(output_size, input_size) * 0.01, 'bias': np.zeros(output_size)},
                {'weight': np.random.randn(output_size, input_size) * 0.01, 'bias': np.zeros(output_size)}
            )
    elif model_type == 'nlinear':
        if feature_extractor.linear is None:
            feature_extractor.set_weights(
                {'weight': np.random.randn(output_size, input_size) * 0.01, 'bias': np.zeros(output_size)}
            )


class ClassifierTrainer(TrainingMixin):
    """
    分类器训练器
    
    Trainer for classification heads.
    """
    
    def __init__(self, n_classes: int, input_dim: int):
        """
        Parameters:
        -----------
        n_classes : int
            类别数
        input_dim : int
            输入特征维度
        """
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.weights = None
        self.training_history = []
    
    def initialize(self, init_type: str = 'xavier'):
        """初始化权重"""
        # For binary classification, output dim is 1 (sigmoid), for multi-class it's n_classes (softmax)
        output_dim = 1 if self.n_classes == 2 else self.n_classes
        self.weights = self._initialize_weights(self.input_dim, output_dim, init_type)
        return self
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        if self.weights is None:
            raise ValueError("Weights not initialized. Call initialize() first.")
        
        logits = x @ self.weights['weight'].T + self.weights['bias']
        
        if self.n_classes == 2:
            # Binary classification: output (batch, 1) with sigmoid
            return self._sigmoid(logits)
        else:
            # Multi-class classification: output (batch, n_classes) with softmax
            return self._softmax(logits)
    
    def train_step(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01
    ) -> float:
        """
        单步训练
        Single training step
        
        Parameters:
        -----------
        x : np.ndarray
            输入特征，shape为 (batch, input_dim)
        targets : np.ndarray
            目标标签，shape为 (batch,) 或 (batch, n_classes)
        learning_rate : float, default=0.01
            学习率
            
        Returns:
        --------
        loss : float
            损失值
        """
        # Forward
        predictions = self.forward(x)
        
        # Convert targets to proper shape
        if self.n_classes == 2:
            # Binary classification: targets should be (batch, 1)
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)
            elif targets.shape[1] > 1:
                # If one-hot encoded, take positive class
                targets = targets[:, 1:2]
        else:
            # Multi-class: convert to one-hot if needed
            if targets.ndim == 1:
                targets_onehot = np.zeros((len(targets), self.n_classes))
                targets_onehot[np.arange(len(targets)), targets.astype(int)] = 1
                targets = targets_onehot
        
        # Compute loss
        loss_type = 'binary_cross_entropy' if self.n_classes == 2 else 'cross_entropy'
        loss = self._compute_loss(predictions, targets, loss_type)
        
        # Compute gradients
        grad_weight, grad_bias = self._compute_gradients_linear(x, predictions, targets, loss_type)
        
        # Update weights
        self.weights = self._update_weights(self.weights, grad_weight, grad_bias, learning_rate)
        
        return loss
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ) -> List[float]:
        """
        训练分类器
        Train classifier
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据特征，shape为 (n_samples, input_dim)
        y : np.ndarray
            训练数据标签，shape为 (n_samples,)
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
        if self.weights is None:
            self.initialize()
        
        n_samples = X.shape[0]
        self.training_history = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                loss = self.train_step(batch_X, batch_y, learning_rate)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            self.training_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.training_history
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """获取权重"""
        return self.weights


class RegressionTrainer(TrainingMixin):
    """
    回归训练器
    
    Trainer for regression heads (forecasting).
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Parameters:
        -----------
        input_dim : int
            输入特征维度
        output_dim : int
            输出维度
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = None
        self.training_history = []
    
    def initialize(self, init_type: str = 'xavier'):
        """初始化权重"""
        self.weights = self._initialize_weights(self.input_dim, self.output_dim, init_type)
        return self
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        if self.weights is None:
            raise ValueError("Weights not initialized. Call initialize() first.")
        
        return x @ self.weights['weight'].T + self.weights['bias']
    
    def train_step(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01,
        loss_type: str = 'mse'
    ) -> float:
        """
        单步训练
        Single training step
        """
        # Forward
        predictions = self.forward(x)
        
        # Compute loss
        loss = self._compute_loss(predictions, targets, loss_type)
        
        # Compute gradients
        grad_weight, grad_bias = self._compute_gradients_linear(x, predictions, targets, loss_type)
        
        # Update weights
        self.weights = self._update_weights(self.weights, grad_weight, grad_bias, learning_rate)
        
        return loss
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        loss_type: str = 'mse',
        verbose: bool = True
    ) -> List[float]:
        """
        训练回归器
        Train regressor
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据特征，shape为 (n_samples, input_dim)
        y : np.ndarray
            训练数据目标，shape为 (n_samples, output_dim)
        epochs : int, default=100
            训练轮数
        learning_rate : float, default=0.01
            学习率
        batch_size : int, default=32
            批大小
        loss_type : str, default='mse'
            损失类型
        verbose : bool, default=True
            是否打印训练信息
            
        Returns:
        --------
        history : List[float]
            训练损失历史
        """
        if self.weights is None:
            self.initialize()
        
        n_samples = X.shape[0]
        self.training_history = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                loss = self.train_step(batch_X, batch_y, learning_rate, loss_type)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            self.training_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.training_history
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """获取权重"""
        return self.weights


def create_training_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    freeze_extractor: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建训练数据
    Create training data by extracting features
    
    Parameters:
    -----------
    X : np.ndarray
        原始输入数据
    y : np.ndarray
        目标数据
    feature_extractor : callable
        特征提取函数
    freeze_extractor : bool, default=True
        是否冻结特征提取器
        
    Returns:
    --------
    features : np.ndarray
        提取的特征
    targets : np.ndarray
        目标数据
    """
    # Extract features
    features = feature_extractor(X)
    
    # Flatten features for linear head
    if features.ndim > 2:
        batch_size = features.shape[0]
        features = features.reshape(batch_size, -1)
    
    return features, y
