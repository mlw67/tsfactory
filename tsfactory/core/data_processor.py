"""
Data Processing Module for Time Series

时序数据处理模块，包括滤波、分解、时序数据转图像等功能
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union
from scipy import signal
from scipy.fft import fft, ifft


class DataProcessor:
    """
    时序数据处理器
    
    提供数据滤波、分解、转换等功能
    """
    
    # 常量定义
    EPSILON = 1e-10  # 用于避免除零的小常数
    
    def __init__(self):
        pass
    
    def filter(
        self,
        data: np.ndarray,
        method: str = 'moving_average',
        window_size: int = 5,
        **kwargs
    ) -> np.ndarray:
        """
        数据滤波处理
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据
        method : str
            滤波方法：'moving_average', 'exponential', 'savgol', 'butterworth'
        window_size : int
            窗口大小
        **kwargs : dict
            其他参数
            
        Returns:
        --------
        filtered_data : np.ndarray
            滤波后的数据
        """
        if data.ndim == 1:
            return self._filter_1d(data, method, window_size, **kwargs)
        else:
            # 多维数据：对每个特征分别滤波
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                result[:, i] = self._filter_1d(data[:, i], method, window_size, **kwargs)
            return result
    
    def _filter_1d(
        self,
        data: np.ndarray,
        method: str,
        window_size: int,
        **kwargs
    ) -> np.ndarray:
        """对一维数据进行滤波"""
        if method == 'moving_average':
            return self._moving_average_filter(data, window_size)
        elif method == 'exponential':
            alpha = kwargs.get('alpha', 0.3)
            return self._exponential_filter(data, alpha)
        elif method == 'savgol':
            polyorder = kwargs.get('polyorder', 2)
            return self._savgol_filter(data, window_size, polyorder)
        elif method == 'butterworth':
            cutoff = kwargs.get('cutoff', 0.1)
            order = kwargs.get('order', 4)
            return self._butterworth_filter(data, cutoff, order)
        else:
            raise ValueError(f"Unknown filter method: {method}")
    
    def _moving_average_filter(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """移动平均滤波"""
        if window_size < 1:
            return data.copy()
        
        # 使用卷积实现移动平均
        kernel = np.ones(window_size) / window_size
        # 使用'same'模式保持输出长度与输入相同
        filtered = np.convolve(data, kernel, mode='same')
        return filtered
    
    def _exponential_filter(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """指数平滑滤波"""
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        
        for i in range(1, len(data)):
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1]
        
        return filtered
    
    def _savgol_filter(self, data: np.ndarray, window_size: int, polyorder: int) -> np.ndarray:
        """Savitzky-Golay滤波"""
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数
        
        if window_size < polyorder + 2:
            window_size = polyorder + 2
            if window_size % 2 == 0:
                window_size += 1
        
        return signal.savgol_filter(data, window_size, polyorder)
    
    def _butterworth_filter(
        self,
        data: np.ndarray,
        cutoff: float,
        order: int
    ) -> np.ndarray:
        """Butterworth低通滤波"""
        # 设计Butterworth滤波器
        b, a = signal.butter(order, cutoff, btype='low', analog=False)
        # 应用滤波器
        filtered = signal.filtfilt(b, a, data)
        return filtered
    
    def decompose(
        self,
        data: np.ndarray,
        method: str = 'additive',
        period: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        时序数据分解
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape为(n_samples,)
        method : str
            分解方法：'additive'（加法模型）或'multiplicative'（乘法模型）
        period : int, optional
            季节周期，如果为None则自动检测
            
        Returns:
        --------
        components : dict
            包含'trend'（趋势）、'seasonal'（季节）、'residual'（残差）的字典
        """
        if data.ndim > 1:
            raise ValueError("Decomposition only supports 1D data")
        
        if period is None:
            period = self._detect_period(data)
        
        # 计算趋势成分（使用移动平均）
        trend = self._extract_trend(data, period)
        
        # 去趋势
        if method == 'additive':
            detrended = data - trend
        else:  # multiplicative
            detrended = data / (trend + self.EPSILON)
        
        # 计算季节成分
        seasonal = self._extract_seasonal(detrended, period)
        
        # 计算残差
        if method == 'additive':
            residual = data - trend - seasonal
        else:  # multiplicative
            residual = data / ((trend + self.EPSILON) * (seasonal + self.EPSILON))
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def _detect_period(self, data: np.ndarray, max_period: Optional[int] = None) -> int:
        """
        自动检测时序数据的周期
        
        使用自相关或FFT方法
        """
        n = len(data)
        if max_period is None:
            max_period = min(n // 2, 50)
        
        # 使用自相关检测周期
        data_centered = data - np.mean(data)
        autocorr = np.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]
        
        # 寻找第一个显著的峰值
        peaks = []
        for lag in range(2, min(max_period, len(autocorr) - 1)):
            if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
                if autocorr[lag] > 0.3:  # 阈值
                    peaks.append((lag, autocorr[lag]))
        
        if peaks:
            # 返回最强的峰值位置
            period = max(peaks, key=lambda x: x[1])[0]
            return period
        
        return 12  # 默认周期
    
    def _extract_trend(self, data: np.ndarray, period: int) -> np.ndarray:
        """提取趋势成分"""
        # 使用移动平均提取趋势
        window_size = period if period % 2 == 1 else period + 1
        return self._moving_average_filter(data, window_size)
    
    def _extract_seasonal(self, detrended: np.ndarray, period: int) -> np.ndarray:
        """提取季节成分"""
        n = len(detrended)
        n_periods = n // period
        
        if n_periods < 2:
            return np.zeros_like(detrended)
        
        # 将数据重塑为多个周期
        seasonal_data = detrended[:n_periods * period].reshape(n_periods, period)
        
        # 计算每个季节位置的平均值
        seasonal_component = np.mean(seasonal_data, axis=0)
        
        # 中心化季节成分
        seasonal_component -= np.mean(seasonal_component)
        
        # 扩展到整个序列
        seasonal = np.tile(seasonal_component, n // period + 1)[:n]
        
        return seasonal
    
    def to_image(
        self,
        data: np.ndarray,
        period: Optional[int] = None,
        norm_min: Optional[float] = None,
        norm_max: Optional[float] = None,
        auto_detect_period: bool = True
    ) -> np.ndarray:
        """
        将时序数据转换为图像
        
        转换规则：
        1. 先归一化（归一化的最大最小值可以自主计算也可以外部传入）
        2. 以周期参数为输入（当周期参数缺失时，自动计算最小周期）
        3. 每个数值作为一个像素点，一个周期为一行进行拼接成2D图
        4. 多特征情况下，每个特征是一个通道
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape为(n_samples,)或(n_samples, n_features)
        period : int, optional
            序列周期
        norm_min : float, optional
            归一化最小值，如果为None则自动计算
        norm_max : float, optional
            归一化最大值，如果为None则自动计算
        auto_detect_period : bool
            当period为None时是否自动检测周期
            
        Returns:
        --------
        image : np.ndarray
            图像数据，shape为(n_rows, period, 2)（单特征，第二通道为1）或(n_rows, period, n_features)（多特征）
        """
        # 确保数据是2D的
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        
        # 归一化
        if norm_min is None:
            norm_min = np.min(data)
        if norm_max is None:
            norm_max = np.max(data)
        
        # 避免除以零
        if norm_max - norm_min < self.EPSILON:
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = (data - norm_min) / (norm_max - norm_min)
        
        # 检测或使用提供的周期
        if period is None:
            if auto_detect_period:
                # 对第一个特征检测周期
                period = self._detect_period(data[:, 0])
            else:
                period = 12  # 默认周期
        
        # 计算能完整容纳的周期数
        n_rows = n_samples // period
        
        if n_rows == 0:
            # 如果数据长度小于周期，将整个数据作为一行
            if n_features == 1:
                image = normalized_data[:, 0].reshape(1, -1)
                # 添加第二个通道，数值为1
                ones_channel = np.ones((1, n_samples))
                return np.stack([image, ones_channel], axis=-1)
            else:
                return normalized_data.reshape(1, n_samples, n_features)
        
        # 截取数据以匹配完整周期
        data_truncated = normalized_data[:n_rows * period]
        
        # 重塑为图像
        if n_features == 1:
            # 单特征：返回(n_rows, period, 2)，第二个通道值为1
            image = data_truncated[:, 0].reshape(n_rows, period)
            # 添加第二个通道，数值为1
            ones_channel = np.ones((n_rows, period))
            image = np.stack([image, ones_channel], axis=-1)
        else:
            # 多特征：返回(n_rows, period, n_features)
            image = data_truncated.reshape(n_rows, period, n_features)
        
        return image
    
    def from_image(
        self,
        image: np.ndarray,
        norm_min: float = 0.0,
        norm_max: float = 1.0
    ) -> np.ndarray:
        """
        将图像转换回时序数据
        
        Parameters:
        -----------
        image : np.ndarray
            图像数据，shape为(n_rows, period, 2)（单特征）或(n_rows, period, n_features)（多特征）
        norm_min : float
            反归一化最小值
        norm_max : float
            反归一化最大值
            
        Returns:
        --------
        data : np.ndarray
            时序数据，shape为(n_samples,)或(n_samples, n_features)
        """
        if image.ndim == 2:
            # 旧版本的单特征图像（已弃用）
            import warnings
            warnings.warn(
                "Support for 2D single-channel images is deprecated. "
                "Please regenerate images using the current version which produces 3D arrays with shape (n_rows, period, 2).",
                DeprecationWarning,
                stacklevel=2
            )
            data = image.flatten()
            # 反归一化
            data = data * (norm_max - norm_min) + norm_min
            return data
        else:
            # 多特征图像
            n_rows, period, n_features = image.shape
            
            # 检查是否是单特征图像（有2个通道，第二个通道全为1）
            # 使用快速检查：只检查第二通道的第一个和最后一个值，以及平均值
            if n_features == 2:
                second_channel = image[:, :, 1]
                # 快速检查：检查最小值、最大值和平均值是否都接近1
                if (abs(second_channel.min() - 1.0) < 1e-6 and 
                    abs(second_channel.max() - 1.0) < 1e-6 and 
                    abs(second_channel.mean() - 1.0) < 1e-6):
                    # 单特征图像，只使用第一个通道
                    data = image[:, :, 0].flatten()
                    # 反归一化
                    data = data * (norm_max - norm_min) + norm_min
                    return data
            
            # 真正的多特征图像
            data = image.reshape(-1, n_features)
            # 反归一化
            data = data * (norm_max - norm_min) + norm_min
            return data
