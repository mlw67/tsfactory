"""
Tests for Data Processor Module

数据处理模块的测试
"""

import numpy as np
import pytest
from tsfactory.core.data_processor import DataProcessor


class TestDataProcessor:
    """测试数据处理器"""
    
    def test_moving_average_filter(self):
        """测试移动平均滤波"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        processor = DataProcessor()
        
        filtered = processor.filter(data, method='moving_average', window_size=3)
        
        assert len(filtered) == len(data)
        # 移动平均应该平滑数据
        assert np.std(filtered) < np.std(data)
    
    def test_exponential_filter(self):
        """测试指数平滑滤波"""
        data = np.random.randn(100)
        processor = DataProcessor()
        
        filtered = processor.filter(data, method='exponential', alpha=0.3)
        
        assert len(filtered) == len(data)
        assert filtered[0] == data[0]  # 第一个值应该相同
    
    def test_savgol_filter(self):
        """测试Savitzky-Golay滤波"""
        data = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1
        processor = DataProcessor()
        
        filtered = processor.filter(data, method='savgol', window_size=11, polyorder=2)
        
        assert len(filtered) == len(data)
    
    def test_butterworth_filter(self):
        """测试Butterworth滤波"""
        data = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1
        processor = DataProcessor()
        
        filtered = processor.filter(data, method='butterworth', cutoff=0.1, order=4)
        
        assert len(filtered) == len(data)
    
    def test_filter_multidimensional(self):
        """测试多维数据滤波"""
        data = np.random.randn(100, 3)
        processor = DataProcessor()
        
        filtered = processor.filter(data, method='moving_average', window_size=5)
        
        assert filtered.shape == data.shape
    
    def test_decompose_additive(self):
        """测试加法模型分解"""
        # 创建带趋势和季节性的数据
        t = np.arange(100)
        trend = 0.05 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.randn(100) * 0.5
        data = trend + seasonal + noise
        
        processor = DataProcessor()
        components = processor.decompose(data, method='additive', period=12)
        
        assert 'trend' in components
        assert 'seasonal' in components
        assert 'residual' in components
        assert len(components['trend']) == len(data)
        assert len(components['seasonal']) == len(data)
        assert len(components['residual']) == len(data)
    
    def test_decompose_multiplicative(self):
        """测试乘法模型分解"""
        # 创建带趋势和季节性的数据
        t = np.arange(100)
        trend = 1 + 0.05 * t
        seasonal = 1 + 0.1 * np.sin(2 * np.pi * t / 12)
        noise = 1 + np.random.randn(100) * 0.05
        data = trend * seasonal * noise
        
        processor = DataProcessor()
        components = processor.decompose(data, method='multiplicative', period=12)
        
        assert 'trend' in components
        assert 'seasonal' in components
        assert 'residual' in components
    
    def test_detect_period(self):
        """测试周期检测"""
        # 创建周期为12的数据
        t = np.arange(100)
        data = 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(100) * 0.5
        
        processor = DataProcessor()
        period = processor._detect_period(data)
        
        # 检测到的周期应该接近12
        assert 10 <= period <= 14
    
    def test_to_image_1d(self):
        """测试一维数据转图像"""
        data = np.arange(60, dtype=float)
        processor = DataProcessor()
        
        image = processor.to_image(data, period=12)
        
        # 应该有5行，每行12个元素，2个通道（第二通道为1）
        assert image.shape == (5, 12, 2)
        # 检查第一通道的归一化
        assert np.min(image[:, :, 0]) >= 0
        assert np.max(image[:, :, 0]) <= 1
        # 检查第二通道全为1
        assert np.allclose(image[:, :, 1], 1.0)
    
    def test_to_image_2d(self):
        """测试多维数据转图像"""
        data = np.random.randn(60, 3)
        processor = DataProcessor()
        
        image = processor.to_image(data, period=10)
        
        # 应该有6行，每行10个元素，3个通道
        assert image.shape == (6, 10, 3)
    
    def test_to_image_with_normalization(self):
        """测试使用指定的归一化参数"""
        data = np.arange(60, dtype=float)
        processor = DataProcessor()
        
        image = processor.to_image(data, period=12, norm_min=0, norm_max=100)
        
        assert image.shape == (5, 12, 2)
        # 第一通道的所有值应该在0-1之间（因为数据范围是0-59，归一化范围是0-100）
        assert np.all(image[:, :, 0] >= 0)
        assert np.all(image[:, :, 0] <= 1)
        # 第二通道全为1
        assert np.allclose(image[:, :, 1], 1.0)
    
    def test_to_image_auto_detect_period(self):
        """测试自动检测周期"""
        # 创建周期为12的数据
        t = np.arange(100)
        data = 10 * np.sin(2 * np.pi * t / 12)
        
        processor = DataProcessor()
        image = processor.to_image(data, period=None, auto_detect_period=True)
        
        # 图像应该成功创建，单通道数据应该有3维（包含第二通道）
        assert image.ndim == 3
        assert image.shape[0] > 0
        assert image.shape[2] == 2  # 两个通道
        assert np.allclose(image[:, :, 1], 1.0)  # 第二通道全为1
    
    def test_from_image_1d(self):
        """测试从图像转回一维数据"""
        # 创建简单的图像
        image = np.arange(60, dtype=float).reshape(5, 12) / 60.0
        processor = DataProcessor()
        
        data = processor.from_image(image, norm_min=0, norm_max=60)
        
        # 数据长度应该是60
        assert len(data) == 60
        # 检查反归一化
        assert np.allclose(data, np.arange(60), atol=0.1)
    
    def test_from_image_2d(self):
        """测试从图像转回多维数据"""
        # 创建多通道图像
        image = np.random.rand(5, 12, 3)
        processor = DataProcessor()
        
        data = processor.from_image(image, norm_min=0, norm_max=1)
        
        # 数据形状应该是(60, 3)
        assert data.shape == (60, 3)
    
    def test_image_roundtrip(self):
        """测试数据->图像->数据的往返转换"""
        original_data = np.arange(60, dtype=float)
        processor = DataProcessor()
        
        # 转换为图像
        image = processor.to_image(original_data, period=12)
        
        # 转换回数据
        recovered_data = processor.from_image(
            image,
            norm_min=np.min(original_data),
            norm_max=np.max(original_data)
        )
        
        # 恢复的数据应该与原始数据接近
        assert np.allclose(original_data, recovered_data, atol=0.1)
    
    def test_to_image_short_data(self):
        """测试数据长度小于周期的情况"""
        data = np.arange(5, dtype=float)
        processor = DataProcessor()
        
        image = processor.to_image(data, period=12)
        
        # 应该创建单行图像，包含2个通道
        assert image.shape == (1, 5, 2)
        assert np.allclose(image[:, :, 1], 1.0)  # 第二通道全为1
    
    def test_single_channel_has_second_channel_as_ones(self):
        """测试单通道图像添加第二通道为1的功能"""
        data = np.random.randn(100)
        processor = DataProcessor()
        
        image = processor.to_image(data, period=20)
        
        # 应该有3个维度
        assert image.ndim == 3
        # 应该有2个通道
        assert image.shape[2] == 2
        # 第二通道应该全为1
        assert np.allclose(image[:, :, 1], 1.0)
        # 第一通道应该是归一化后的数据
        assert image.shape == (5, 20, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
