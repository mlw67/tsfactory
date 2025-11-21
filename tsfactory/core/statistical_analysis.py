"""
Statistical Analysis Module for Time Series

时序数据的统计分析模块
"""

import numpy as np
from typing import Dict, Optional, Union
from scipy import stats


class StatisticalAnalyzer:
    """
    时序统计分析器
    
    提供各种时序数据的统计分析功能
    """
    
    def __init__(self):
        pass
    
    def descriptive_statistics(self, data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        计算描述性统计量
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape可以是(n_samples,)或(n_samples, n_features)
            
        Returns:
        --------
        stats : dict
            包含各种统计量的字典
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        result = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'var': np.var(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0),
            'skewness': stats.skew(data, axis=0),
            'kurtosis': stats.kurtosis(data, axis=0),
        }
        
        return result
    
    def autocorrelation(
        self, 
        data: np.ndarray, 
        max_lag: Optional[int] = None
    ) -> np.ndarray:
        """
        计算自相关函数 (ACF)
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape为(n_samples,)
        max_lag : int, optional
            最大滞后阶数，默认为min(n_samples//2, 40)
            
        Returns:
        --------
        acf : np.ndarray
            自相关系数数组
        """
        if data.ndim > 1:
            raise ValueError("Autocorrelation only supports 1D data")
        
        n = len(data)
        if max_lag is None:
            max_lag = min(n // 2, 40)
        
        data_centered = data - np.mean(data)
        variance = np.var(data)
        
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        
        for lag in range(1, max_lag + 1):
            acf[lag] = np.sum(data_centered[:-lag] * data_centered[lag:]) / (n * variance)
        
        return acf
    
    def partial_autocorrelation(
        self, 
        data: np.ndarray, 
        max_lag: Optional[int] = None
    ) -> np.ndarray:
        """
        计算偏自相关函数 (PACF)
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape为(n_samples,)
        max_lag : int, optional
            最大滞后阶数
            
        Returns:
        --------
        pacf : np.ndarray
            偏自相关系数数组
        """
        if data.ndim > 1:
            raise ValueError("Partial autocorrelation only supports 1D data")
        
        n = len(data)
        if max_lag is None:
            max_lag = min(n // 2, 40)
        
        acf_values = self.autocorrelation(data, max_lag)
        pacf = np.zeros(max_lag + 1)
        pacf[0] = 1.0
        
        if max_lag > 0:
            pacf[1] = acf_values[1]
        
        for k in range(2, max_lag + 1):
            # Levinson-Durbin递归算法
            numerator = acf_values[k] - np.sum(
                pacf[1:k] * acf_values[k-1:0:-1]
            )
            denominator = 1 - np.sum(
                pacf[1:k] * acf_values[1:k]
            )
            pacf[k] = numerator / denominator
        
        return pacf
    
    def trend_analysis(self, data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        趋势分析
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape为(n_samples,)或(n_samples, n_features)
            
        Returns:
        --------
        trend_info : dict
            包含趋势信息的字典
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        x = np.arange(n_samples).reshape(-1, 1)
        
        slopes = []
        intercepts = []
        r_squared = []
        
        for i in range(n_features):
            y = data[:, i]
            # 线性回归
            slope, intercept, r_value, _, _ = stats.linregress(x.flatten(), y)
            slopes.append(slope)
            intercepts.append(intercept)
            r_squared.append(r_value ** 2)
        
        return {
            'slope': np.array(slopes),
            'intercept': np.array(intercepts),
            'r_squared': np.array(r_squared),
        }
    
    def seasonality_strength(
        self, 
        data: np.ndarray, 
        period: int
    ) -> float:
        """
        计算季节性强度
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape为(n_samples,)
        period : int
            季节周期
            
        Returns:
        --------
        strength : float
            季节性强度 (0-1之间)
        """
        if data.ndim > 1:
            raise ValueError("Seasonality strength only supports 1D data")
        
        n = len(data)
        if n < 2 * period:
            return 0.0
        
        # 计算季节性成分
        n_periods = n // period
        seasonal_data = data[:n_periods * period].reshape(n_periods, period)
        seasonal_component = np.mean(seasonal_data, axis=0)
        seasonal_component = np.tile(seasonal_component, n_periods)
        
        # 计算残差
        residual = data[:n_periods * period] - seasonal_component
        
        # 季节性强度 = 1 - Var(residual) / Var(data)
        var_residual = np.var(residual)
        var_data = np.var(data[:n_periods * period])
        
        if var_data == 0:
            return 0.0
        
        strength = max(0, 1 - var_residual / var_data)
        return strength
