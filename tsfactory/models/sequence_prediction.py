"""
Sequence Prediction Module for Time Series

时序数据的序列预测模块
"""

import numpy as np
from typing import Optional, Tuple, Union


class SequencePredictor:
    """
    序列预测器
    
    支持多种预测模型，适用于单维度和多维度输入/输出
    """
    
    def __init__(self, model_type: str = 'ar', **kwargs):
        """
        Parameters:
        -----------
        model_type : str
            模型类型：'ar', 'ma', 'arma', 'linear', 'polynomial'
        **kwargs : dict
            模型参数
        """
        self.model_type = model_type
        self.model_params = kwargs
        self.is_fitted = False
        self.coef_ = None
        self.input_dim = None
        self.output_dim = None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        order: Optional[int] = None
    ) -> 'SequencePredictor':
        """
        训练预测模型
        
        Parameters:
        -----------
        X : np.ndarray
            输入数据，shape为(n_samples, n_features)或(n_samples,)
        y : np.ndarray, optional
            目标数据，如果为None则从X构建滞后数据
        order : int, optional
            滞后阶数（仅用于自回归模型）
            
        Returns:
        --------
        self : SequencePredictor
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y is None:
            # 自回归：从X构建训练数据
            if order is None:
                order = self.model_params.get('order', 1)
            X, y = self._prepare_autoregressive_data(X, order)
        else:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
        
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1] if y.ndim > 1 else 1
        
        if self.model_type == 'ar':
            self._fit_ar(X, y)
        elif self.model_type == 'linear':
            self._fit_linear(X, y)
        elif self.model_type == 'polynomial':
            degree = self.model_params.get('degree', 2)
            self._fit_polynomial(X, y, degree)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self, 
        X: Optional[np.ndarray] = None, 
        steps: int = 1,
        last_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        进行预测
        
        Parameters:
        -----------
        X : np.ndarray, optional
            输入数据
        steps : int
            预测步数（用于多步预测）
        last_values : np.ndarray, optional
            用于自回归预测的最后几个值
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if X is None:
            # 自回归多步预测
            if last_values is None:
                raise ValueError("Either X or last_values must be provided")
            return self._predict_multi_step(last_values, steps)
        else:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            # 处理线性模型的截距
            if self.model_type == 'linear':
                X = np.column_stack([np.ones(len(X)), X])
            
            if self.model_type == 'polynomial':
                degree = self.model_params.get('degree', 2)
                X = self._polynomial_features(X, degree)
            
            predictions = X @ self.coef_
            
            if self.output_dim == 1 and predictions.ndim > 1:
                predictions = predictions.flatten()
            
            return predictions
    
    def _fit_ar(self, X: np.ndarray, y: np.ndarray):
        """拟合自回归模型"""
        # 使用OLS估计
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
    
    def _fit_linear(self, X: np.ndarray, y: np.ndarray):
        """拟合线性回归模型"""
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        self.coef_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        self.has_intercept = True
    
    def _fit_polynomial(self, X: np.ndarray, y: np.ndarray, degree: int):
        """拟合多项式回归模型"""
        X_poly = self._polynomial_features(X, degree)
        self.coef_ = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    
    def _polynomial_features(self, X: np.ndarray, degree: int) -> np.ndarray:
        """
        生成多项式特征
        
        对于一维输入，生成 [1, x, x^2, ..., x^degree]
        """
        n_samples = X.shape[0]
        
        if X.shape[1] == 1:
            # 一维情况
            features = [np.ones((n_samples, 1))]
            for d in range(1, degree + 1):
                features.append(X ** d)
            return np.hstack(features)
        else:
            # 多维情况（简化版，只包含单变量的幂次）
            features = [np.ones((n_samples, 1))]
            for d in range(1, degree + 1):
                features.append(X ** d)
            return np.hstack(features)
    
    def _prepare_autoregressive_data(
        self, 
        data: np.ndarray, 
        order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备自回归数据
        
        构建滞后特征矩阵X和目标向量y
        """
        n_samples, n_features = data.shape
        
        if n_samples <= order:
            raise ValueError(f"Data length ({n_samples}) must be greater than order ({order})")
        
        # 构建滞后特征
        X = []
        for i in range(order):
            X.append(data[order - i - 1:n_samples - i - 1])
        
        X = np.hstack(X)
        y = data[order:]
        
        return X, y
    
    def _predict_multi_step(
        self, 
        last_values: np.ndarray, 
        steps: int
    ) -> np.ndarray:
        """
        多步预测（自回归）
        
        使用递归策略进行多步预测
        """
        if last_values.ndim == 1:
            last_values = last_values.reshape(-1, 1)
        
        order = len(last_values)
        n_features = last_values.shape[1]
        
        predictions = []
        current_window = last_values.copy()
        
        for _ in range(steps):
            # 准备输入
            X_input = current_window.flatten().reshape(1, -1)
            
            # 预测下一步
            next_pred = X_input @ self.coef_
            
            if next_pred.ndim == 1:
                next_pred = next_pred.reshape(1, -1)
            
            predictions.append(next_pred[0])
            
            # 更新窗口
            current_window = np.vstack([current_window[1:], next_pred])
        
        predictions = np.array(predictions)
        
        if n_features == 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> dict:
        """
        评估模型性能
        
        Parameters:
        -----------
        X : np.ndarray
            输入数据
        y : np.ndarray
            真实目标值
            
        Returns:
        --------
        metrics : dict
            评估指标
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # 计算各种指标
        mse = np.mean((y - y_pred) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred), axis=0)
        
        # R^2
        ss_res = np.sum((y - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        return {
            'mse': float(mse[0]) if len(mse) == 1 else mse.tolist(),
            'rmse': float(rmse[0]) if len(rmse) == 1 else rmse.tolist(),
            'mae': float(mae[0]) if len(mae) == 1 else mae.tolist(),
            'r2': float(r2[0]) if len(r2) == 1 else r2.tolist()
        }
