"""
Hypothesis Testing Module for Time Series

时序数据的假设检验模块
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats


class HypothesisTester:
    """
    时序假设检验器
    
    提供各种统计假设检验功能
    """
    
    def __init__(self):
        pass
    
    def stationarity_test(
        self, 
        data: np.ndarray, 
        method: str = 'adf'
    ) -> Dict[str, float]:
        """
        平稳性检验 (ADF检验或KPSS检验)
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape为(n_samples,)
        method : str
            检验方法，'adf'或'kpss'
            
        Returns:
        --------
        result : dict
            检验结果
        """
        if data.ndim > 1:
            raise ValueError("Stationarity test only supports 1D data")
        
        if method == 'adf':
            return self._adf_test(data)
        elif method == 'kpss':
            return self._kpss_test(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _adf_test(self, data: np.ndarray) -> Dict[str, float]:
        """
        增强迪基-福勒检验 (Augmented Dickey-Fuller Test)
        
        H0: 序列存在单位根（非平稳）
        H1: 序列不存在单位根（平稳）
        """
        n = len(data)
        max_lag = int(np.floor(12 * (n / 100) ** 0.25))
        
        # 简化版ADF检验实现
        data_diff = np.diff(data)
        data_lag = data[:-1]
        
        # 构建回归模型：Δy_t = α + βy_{t-1} + ε_t
        X = np.column_stack([np.ones(len(data_lag)), data_lag])
        y = data_diff
        
        # OLS估计
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        residuals = y - y_pred
        
        # 计算t统计量
        rss = np.sum(residuals ** 2)
        var_beta = rss / (len(y) - 2) * np.linalg.inv(X.T @ X)
        t_stat = beta[1] / np.sqrt(var_beta[1, 1])
        
        # 临界值（简化版）
        critical_values = {
            '1%': -3.43,
            '5%': -2.86,
            '10%': -2.57
        }
        
        # p值近似
        p_value = 0.05 if t_stat > critical_values['5%'] else 0.01
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': t_stat < critical_values['5%']
        }
    
    def _kpss_test(self, data: np.ndarray) -> Dict[str, float]:
        """
        KPSS检验
        
        H0: 序列是平稳的
        H1: 序列是非平稳的
        """
        n = len(data)
        
        # 计算趋势
        t = np.arange(n)
        X = np.column_stack([np.ones(n), t])
        beta = np.linalg.lstsq(X, data, rcond=None)[0]
        residuals = data - X @ beta
        
        # 计算部分和
        S = np.cumsum(residuals)
        
        # 计算长期方差
        s_squared = np.var(residuals)
        
        # KPSS统计量
        kpss_stat = np.sum(S ** 2) / (n ** 2 * s_squared)
        
        # 临界值
        critical_values = {
            '1%': 0.216,
            '5%': 0.146,
            '10%': 0.119
        }
        
        p_value = 0.05 if kpss_stat > critical_values['5%'] else 0.1
        
        return {
            'test_statistic': kpss_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': kpss_stat < critical_values['5%']
        }
    
    def normality_test(self, data: np.ndarray) -> Dict[str, float]:
        """
        正态性检验 (Shapiro-Wilk检验)
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
            
        Returns:
        --------
        result : dict
            检验结果
        """
        if data.ndim > 1:
            data = data.flatten()
        
        statistic, p_value = stats.shapiro(data)
        
        return {
            'test_statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def correlation_test(
        self, 
        data1: np.ndarray, 
        data2: np.ndarray
    ) -> Dict[str, float]:
        """
        相关性检验 (Pearson相关检验)
        
        Parameters:
        -----------
        data1 : np.ndarray
            第一个时序数据
        data2 : np.ndarray
            第二个时序数据
            
        Returns:
        --------
        result : dict
            检验结果
        """
        if data1.ndim > 1:
            data1 = data1.flatten()
        if data2.ndim > 1:
            data2 = data2.flatten()
        
        correlation, p_value = stats.pearsonr(data1, data2)
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
    
    def white_noise_test(
        self, 
        data: np.ndarray, 
        lags: int = 10
    ) -> Dict[str, float]:
        """
        白噪声检验 (Ljung-Box检验)
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
        lags : int
            滞后阶数
            
        Returns:
        --------
        result : dict
            检验结果
        """
        if data.ndim > 1:
            data = data.flatten()
        
        n = len(data)
        
        # 计算自相关系数
        data_centered = data - np.mean(data)
        variance = np.var(data)
        
        acf = []
        for lag in range(1, lags + 1):
            acf_lag = np.sum(data_centered[:-lag] * data_centered[lag:]) / (n * variance)
            acf.append(acf_lag)
        
        # Ljung-Box统计量
        lb_stat = n * (n + 2) * np.sum(np.array(acf) ** 2 / (n - np.arange(1, lags + 1)))
        
        # 自由度为lags
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        return {
            'test_statistic': lb_stat,
            'p_value': p_value,
            'is_white_noise': p_value > 0.05
        }
