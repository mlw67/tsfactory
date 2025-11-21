"""
Missing Value Imputation Module for Time Series

时序数据的缺失值填充模块
"""

import numpy as np
from typing import Optional, Union
from scipy.interpolate import interp1d


class MissingValueImputer:
    """
    缺失值填充器
    
    支持多种缺失值填充方法，适用于单维度和多维度时序数据
    """
    
    def __init__(self, method: str = 'linear'):
        """
        Parameters:
        -----------
        method : str
            填充方法：'linear', 'forward', 'backward', 'mean', 'median', 'spline'
        """
        self.method = method
    
    def fit_transform(
        self, 
        data: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        填充缺失值
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，缺失值用np.nan表示
        mask : np.ndarray, optional
            缺失值掩码，True表示缺失
            
        Returns:
        --------
        filled_data : np.ndarray
            填充后的数据
        """
        if data.ndim == 1:
            return self._impute_1d(data, mask)
        else:
            # 多维数据：对每个特征分别填充
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                col_mask = mask[:, i] if mask is not None else None
                result[:, i] = self._impute_1d(data[:, i], col_mask)
            return result
    
    def _impute_1d(
        self, 
        data: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        对一维数据进行缺失值填充
        """
        if mask is None:
            mask = np.isnan(data)
        
        if not np.any(mask):
            return data.copy()
        
        filled_data = data.copy()
        
        if self.method == 'forward':
            filled_data = self._forward_fill(filled_data, mask)
        elif self.method == 'backward':
            filled_data = self._backward_fill(filled_data, mask)
        elif self.method == 'mean':
            filled_data = self._mean_fill(filled_data, mask)
        elif self.method == 'median':
            filled_data = self._median_fill(filled_data, mask)
        elif self.method == 'linear':
            filled_data = self._linear_interpolate(filled_data, mask)
        elif self.method == 'spline':
            filled_data = self._spline_interpolate(filled_data, mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return filled_data
    
    def _forward_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """前向填充"""
        result = data.copy()
        for i in range(len(data)):
            if mask[i]:
                if i > 0:
                    result[i] = result[i - 1]
        return result
    
    def _backward_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """后向填充"""
        result = data.copy()
        for i in range(len(data) - 1, -1, -1):
            if mask[i]:
                if i < len(data) - 1:
                    result[i] = result[i + 1]
        return result
    
    def _mean_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """均值填充"""
        result = data.copy()
        mean_value = np.nanmean(data)
        result[mask] = mean_value
        return result
    
    def _median_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """中位数填充"""
        result = data.copy()
        median_value = np.nanmedian(data)
        result[mask] = median_value
        return result
    
    def _linear_interpolate(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """线性插值"""
        result = data.copy()
        valid_indices = np.where(~mask)[0]
        
        if len(valid_indices) < 2:
            # 如果有效值少于2个，使用均值填充
            return self._mean_fill(data, mask)
        
        valid_values = data[valid_indices]
        
        # 使用scipy的线性插值
        f = interp1d(
            valid_indices, 
            valid_values, 
            kind='linear', 
            fill_value='extrapolate'
        )
        
        missing_indices = np.where(mask)[0]
        result[missing_indices] = f(missing_indices)
        
        return result
    
    def _spline_interpolate(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """样条插值"""
        result = data.copy()
        valid_indices = np.where(~mask)[0]
        
        if len(valid_indices) < 4:
            # 样条插值至少需要4个点，否则使用线性插值
            return self._linear_interpolate(data, mask)
        
        valid_values = data[valid_indices]
        
        # 使用cubic样条插值
        f = interp1d(
            valid_indices, 
            valid_values, 
            kind='cubic', 
            fill_value='extrapolate'
        )
        
        missing_indices = np.where(mask)[0]
        result[missing_indices] = f(missing_indices)
        
        return result
    
    def detect_missing(self, data: np.ndarray) -> np.ndarray:
        """
        检测缺失值位置
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
            
        Returns:
        --------
        mask : np.ndarray
            缺失值掩码，True表示缺失
        """
        return np.isnan(data)
    
    def missing_statistics(self, data: np.ndarray) -> dict:
        """
        计算缺失值统计信息
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
            
        Returns:
        --------
        stats : dict
            缺失值统计信息
        """
        mask = self.detect_missing(data)
        
        total_values = data.size
        missing_count = np.sum(mask)
        missing_ratio = missing_count / total_values
        
        if data.ndim > 1:
            # 多维数据：计算每个特征的缺失情况
            missing_per_feature = np.sum(mask, axis=0)
            return {
                'total_missing': missing_count,
                'missing_ratio': missing_ratio,
                'missing_per_feature': missing_per_feature
            }
        else:
            return {
                'total_missing': missing_count,
                'missing_ratio': missing_ratio
            }
