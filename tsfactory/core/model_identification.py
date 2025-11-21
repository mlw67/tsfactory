"""
Model Identification Module for Time Series

时序模型辨识模块
"""

import numpy as np
from typing import Dict, Tuple, Optional


class ModelIdentifier:
    """
    时序模型辨识器
    
    用于识别适合的时序模型（如ARIMA模型的阶数）
    """
    
    def __init__(self):
        pass
    
    def identify_ar_order(
        self, 
        data: np.ndarray, 
        max_order: int = 10,
        criterion: str = 'aic'
    ) -> Dict[str, any]:
        """
        识别AR模型阶数
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape为(n_samples,)
        max_order : int
            最大阶数
        criterion : str
            信息准则，'aic'或'bic'
            
        Returns:
        --------
        result : dict
            包含最优阶数和信息准则值的字典
        """
        if data.ndim > 1:
            raise ValueError("AR order identification only supports 1D data")
        
        n = len(data)
        ic_values = []
        
        for p in range(1, max_order + 1):
            # 拟合AR(p)模型
            X, y = self._prepare_ar_data(data, p)
            
            if len(y) == 0:
                break
            
            # OLS估计
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ coef
                residuals = y - y_pred
                rss = np.sum(residuals ** 2)
                
                # 计算信息准则
                if criterion == 'aic':
                    ic = self._calculate_aic(rss, len(y), p)
                else:  # bic
                    ic = self._calculate_bic(rss, len(y), p)
                
                ic_values.append(ic)
            except (np.linalg.LinAlgError, ValueError):
                ic_values.append(np.inf)
        
        if not ic_values:
            return {'order': 0, 'criterion_value': np.inf}
        
        optimal_order = np.argmin(ic_values) + 1
        
        return {
            'order': optimal_order,
            'criterion_value': ic_values[optimal_order - 1],
            'all_values': ic_values
        }
    
    def identify_ma_order(
        self, 
        data: np.ndarray, 
        max_order: int = 10,
        criterion: str = 'aic'
    ) -> Dict[str, any]:
        """
        识别MA模型阶数（简化版）
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
        max_order : int
            最大阶数
        criterion : str
            信息准则
            
        Returns:
        --------
        result : dict
            包含最优阶数的字典
        """
        if data.ndim > 1:
            raise ValueError("MA order identification only supports 1D data")
        
        # 简化实现：基于PACF截尾特性
        from .statistical_analysis import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer()
        pacf = analyzer.partial_autocorrelation(data, max_order)
        
        # 找到PACF显著性阈值
        n = len(data)
        threshold = 1.96 / np.sqrt(n)
        
        # 找到第一个不显著的PACF值
        for q in range(1, len(pacf)):
            if abs(pacf[q]) < threshold:
                return {'order': q - 1, 'criterion_value': 0.0}
        
        return {'order': max_order, 'criterion_value': 0.0}
    
    def identify_arima_order(
        self, 
        data: np.ndarray, 
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        criterion: str = 'aic'
    ) -> Dict[str, any]:
        """
        识别ARIMA(p,d,q)模型阶数
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
        max_p : int
            AR最大阶数
        max_d : int
            差分最大阶数
        max_q : int
            MA最大阶数
        criterion : str
            信息准则
            
        Returns:
        --------
        result : dict
            包含最优(p,d,q)的字典
        """
        if data.ndim > 1:
            raise ValueError("ARIMA order identification only supports 1D data")
        
        # 首先确定差分阶数d
        d = self._identify_d(data, max_d)
        
        # 对差分后的数据识别p和q
        data_diff = data.copy()
        for _ in range(d):
            data_diff = np.diff(data_diff)
        
        # 识别AR阶数
        ar_result = self.identify_ar_order(data_diff, max_p, criterion)
        p = ar_result['order']
        
        # 识别MA阶数（简化）
        ma_result = self.identify_ma_order(data_diff, max_q, criterion)
        q = ma_result['order']
        
        return {
            'p': p,
            'd': d,
            'q': q,
            'order': (p, d, q)
        }
    
    def _identify_d(self, data: np.ndarray, max_d: int) -> int:
        """
        识别差分阶数
        
        使用ADF检验确定需要差分几次才能使序列平稳
        """
        from .hypothesis_testing import HypothesisTester
        
        tester = HypothesisTester()
        
        for d in range(max_d + 1):
            data_diff = data.copy()
            for _ in range(d):
                data_diff = np.diff(data_diff)
            
            # ADF检验
            result = tester.stationarity_test(data_diff, method='adf')
            if result['is_stationary']:
                return d
        
        return max_d
    
    def _prepare_ar_data(
        self, 
        data: np.ndarray, 
        order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备AR模型的数据
        
        构建X矩阵和y向量
        """
        n = len(data)
        if n <= order:
            return np.array([]), np.array([])
        
        X = np.zeros((n - order, order))
        for i in range(order):
            X[:, i] = data[order - i - 1:n - i - 1]
        
        y = data[order:]
        
        return X, y
    
    def _calculate_aic(self, rss: float, n: int, k: int) -> float:
        """计算AIC"""
        return n * np.log(rss / n) + 2 * k
    
    def _calculate_bic(self, rss: float, n: int, k: int) -> float:
        """计算BIC"""
        return n * np.log(rss / n) + k * np.log(n)
