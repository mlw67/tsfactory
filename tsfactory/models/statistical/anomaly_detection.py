"""
Anomaly Detection Module for Time Series

时序数据的异常检测模块
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from scipy import stats


class AnomalyDetector:
    """
    异常检测器
    
    支持多种异常检测方法，适用于单维度和多维度时序数据
    """
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0):
        """
        Parameters:
        -----------
        method : str
            检测方法：'zscore', 'iqr', 'isolation', 'moving_average'
        threshold : float
            异常阈值
        """
        self.method = method
        self.threshold = threshold
    
    def detect(
        self, 
        data: np.ndarray,
        return_scores: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        检测异常点
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
        return_scores : bool
            是否返回异常分数
            
        Returns:
        --------
        anomalies : np.ndarray
            异常点掩码，True表示异常
        scores : np.ndarray, optional
            异常分数（如果return_scores=True）
        """
        if data.ndim == 1:
            anomalies, scores = self._detect_1d(data)
        else:
            # 多维数据：对每个特征分别检测
            anomalies = np.zeros(data.shape, dtype=bool)
            scores = np.zeros(data.shape)
            for i in range(data.shape[1]):
                anomalies[:, i], scores[:, i] = self._detect_1d(data[:, i])
        
        if return_scores:
            return anomalies, scores
        return anomalies
    
    def _detect_1d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对一维数据进行异常检测
        """
        if self.method == 'zscore':
            return self._zscore_detection(data)
        elif self.method == 'iqr':
            return self._iqr_detection(data)
        elif self.method == 'isolation':
            return self._isolation_detection(data)
        elif self.method == 'moving_average':
            return self._moving_average_detection(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _zscore_detection(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Z-score异常检测
        
        异常点定义为距离均值超过threshold个标准差的点
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            scores = np.zeros_like(data)
            anomalies = np.zeros_like(data, dtype=bool)
        else:
            scores = np.abs((data - mean) / std)
            anomalies = scores > self.threshold
        
        return anomalies, scores
    
    def _iqr_detection(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        IQR (四分位距) 异常检测
        
        异常点定义为超出[Q1-threshold*IQR, Q3+threshold*IQR]范围的点
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        
        anomalies = (data < lower_bound) | (data > upper_bound)
        
        # 计算异常分数（距离边界的距离）
        scores = np.zeros_like(data)
        scores[data < lower_bound] = (lower_bound - data[data < lower_bound]) / iqr
        scores[data > upper_bound] = (data[data > upper_bound] - upper_bound) / iqr
        
        return anomalies, scores
    
    def _isolation_detection(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于孤立森林思想的简化异常检测
        
        使用局部密度估计
        """
        n = len(data)
        k = min(10, n // 2)  # 邻居数量
        
        scores = np.zeros(n)
        
        for i in range(n):
            # 计算到最近k个点的平均距离
            distances = np.abs(data - data[i])
            knn_distances = np.partition(distances, k)[:k+1]
            avg_distance = np.mean(knn_distances[1:])  # 排除自己
            scores[i] = avg_distance
        
        # 标准化分数
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score > 0:
            normalized_scores = (scores - mean_score) / std_score
        else:
            normalized_scores = scores
        
        anomalies = normalized_scores > self.threshold
        
        return anomalies, normalized_scores
    
    def _moving_average_detection(
        self, 
        data: np.ndarray, 
        window: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于移动平均的异常检测
        
        异常点定义为与移动平均的偏差超过threshold个标准差的点
        """
        n = len(data)
        window = min(window, n // 2)
        
        # 计算移动平均
        ma = np.zeros(n)
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            ma[i] = np.mean(data[start:end])
        
        # 计算残差
        residuals = data - ma
        
        # 计算残差的标准差
        std_residual = np.std(residuals)
        
        if std_residual == 0:
            scores = np.zeros_like(data)
            anomalies = np.zeros_like(data, dtype=bool)
        else:
            scores = np.abs(residuals / std_residual)
            anomalies = scores > self.threshold
        
        return anomalies, scores
    
    def get_anomaly_indices(self, data: np.ndarray) -> np.ndarray:
        """
        获取异常点的索引
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
            
        Returns:
        --------
        indices : np.ndarray
            异常点的索引数组
        """
        anomalies = self.detect(data)
        
        if data.ndim == 1:
            return np.where(anomalies)[0]
        else:
            # 多维数据：返回任意特征有异常的样本索引
            return np.where(np.any(anomalies, axis=1))[0]
    
    def anomaly_statistics(self, data: np.ndarray) -> Dict[str, any]:
        """
        计算异常统计信息
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
            
        Returns:
        --------
        stats : dict
            异常统计信息
        """
        anomalies = self.detect(data)
        
        total_values = data.size if data.ndim == 1 else data.shape[0]
        
        if data.ndim == 1:
            anomaly_count = np.sum(anomalies)
            anomaly_ratio = anomaly_count / total_values
            
            return {
                'total_anomalies': int(anomaly_count),
                'anomaly_ratio': float(anomaly_ratio),
                'anomaly_indices': np.where(anomalies)[0].tolist()
            }
        else:
            # 多维数据
            anomaly_per_sample = np.any(anomalies, axis=1)
            anomaly_count = np.sum(anomaly_per_sample)
            anomaly_ratio = anomaly_count / total_values
            
            anomalies_per_feature = np.sum(anomalies, axis=0)
            
            return {
                'total_anomalies': int(anomaly_count),
                'anomaly_ratio': float(anomaly_ratio),
                'anomaly_indices': np.where(anomaly_per_sample)[0].tolist(),
                'anomalies_per_feature': anomalies_per_feature.tolist()
            }
