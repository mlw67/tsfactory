"""
Data Processing Example: Filtering, Decomposition, and Time Series to Image Conversion

数据处理示例：滤波、分解、时序转图像
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tsfactory import DataProcessor


def main():
    print("=" * 70)
    print("TSFactory Data Processing Example - 数据处理示例")
    print("=" * 70)
    
    # 1. 数据滤波 - Data Filtering
    print("\n1. Data Filtering - 数据滤波")
    print("-" * 70)
    
    # 创建带噪声的数据
    t = np.arange(100)
    clean_signal = 10 * np.sin(2 * np.pi * t / 12)
    noisy_signal = clean_signal + np.random.randn(100) * 2
    
    processor = DataProcessor()
    
    # 移动平均滤波
    ma_filtered = processor.filter(noisy_signal, method='moving_average', window_size=5)
    print(f"移动平均滤波 - 原始信号标准差: {np.std(noisy_signal):.4f}")
    print(f"移动平均滤波 - 滤波后标准差: {np.std(ma_filtered):.4f}")
    
    # 指数平滑滤波
    exp_filtered = processor.filter(noisy_signal, method='exponential', alpha=0.3)
    print(f"指数平滑滤波 - 滤波后标准差: {np.std(exp_filtered):.4f}")
    
    # Savitzky-Golay滤波
    savgol_filtered = processor.filter(noisy_signal, method='savgol', window_size=11, polyorder=2)
    print(f"Savitzky-Golay滤波 - 滤波后标准差: {np.std(savgol_filtered):.4f}")
    
    # Butterworth滤波
    butter_filtered = processor.filter(noisy_signal, method='butterworth', cutoff=0.1, order=4)
    print(f"Butterworth滤波 - 滤波后标准差: {np.std(butter_filtered):.4f}")
    
    # 多维数据滤波
    multi_data = np.column_stack([noisy_signal, noisy_signal + 5, noisy_signal - 5])
    multi_filtered = processor.filter(multi_data, method='moving_average', window_size=5)
    print(f"\n多维数据滤波 - 输入形状: {multi_data.shape}")
    print(f"多维数据滤波 - 输出形状: {multi_filtered.shape}")
    
    # 2. 时序分解 - Time Series Decomposition
    print("\n2. Time Series Decomposition - 时序分解")
    print("-" * 70)
    
    # 创建带趋势和季节性的数据
    t = np.arange(200)
    trend = 0.05 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(200) * 0.5
    data_with_trend = trend + seasonal + noise
    
    # 加法模型分解
    components = processor.decompose(data_with_trend, method='additive', period=12)
    
    print("加法模型分解结果:")
    print(f"  趋势成分 - 均值: {np.mean(components['trend']):.4f}")
    print(f"  季节成分 - 均值: {np.mean(components['seasonal']):.4f}")
    print(f"  季节成分 - 标准差: {np.std(components['seasonal']):.4f}")
    print(f"  残差成分 - 标准差: {np.std(components['residual']):.4f}")
    
    # 乘法模型分解
    trend_mult = 1 + 0.05 * t
    seasonal_mult = 1 + 0.1 * np.sin(2 * np.pi * t / 12)
    noise_mult = 1 + np.random.randn(200) * 0.05
    data_mult = trend_mult * seasonal_mult * noise_mult
    
    components_mult = processor.decompose(data_mult, method='multiplicative', period=12)
    print("\n乘法模型分解结果:")
    print(f"  趋势成分 - 均值: {np.mean(components_mult['trend']):.4f}")
    print(f"  残差成分 - 标准差: {np.std(components_mult['residual']):.4f}")
    
    # 自动检测周期
    periodic_data = 10 * np.sin(2 * np.pi * np.arange(100) / 15)
    auto_components = processor.decompose(periodic_data, method='additive', period=None)
    print(f"\n自动周期检测 - 分解成功，残差标准差: {np.std(auto_components['residual']):.4f}")
    
    # 3. 时序数据转图像 - Time Series to Image
    print("\n3. Time Series to Image Conversion - 时序数据转图像")
    print("-" * 70)
    
    # 单维度数据转图像
    data_1d = np.arange(60, dtype=float)
    image_1d = processor.to_image(data_1d, period=12)
    
    print(f"单维数据转图像:")
    print(f"  输入数据形状: {data_1d.shape}")
    print(f"  输出图像形状: {image_1d.shape} (5行 × 12列)")
    print(f"  图像值范围: [{np.min(image_1d):.4f}, {np.max(image_1d):.4f}]")
    
    # 多维度数据转图像（多通道）
    data_2d = np.random.randn(60, 3) * 10 + 50
    image_2d = processor.to_image(data_2d, period=10)
    
    print(f"\n多维数据转图像（多通道）:")
    print(f"  输入数据形状: {data_2d.shape}")
    print(f"  输出图像形状: {image_2d.shape} (6行 × 10列 × 3通道)")
    print(f"  图像值范围: [{np.min(image_2d):.4f}, {np.max(image_2d):.4f}]")
    
    # 指定归一化范围
    image_custom_norm = processor.to_image(
        data_1d, 
        period=12, 
        norm_min=0, 
        norm_max=100
    )
    print(f"\n自定义归一化范围:")
    print(f"  归一化范围: [0, 100]")
    print(f"  图像形状: {image_custom_norm.shape}")
    print(f"  图像值范围: [{np.min(image_custom_norm):.4f}, {np.max(image_custom_norm):.4f}]")
    
    # 自动检测周期
    periodic_signal = 10 * np.sin(2 * np.pi * np.arange(100) / 12)
    image_auto_period = processor.to_image(
        periodic_signal, 
        period=None, 
        auto_detect_period=True
    )
    print(f"\n自动检测周期:")
    print(f"  输入数据长度: {len(periodic_signal)}")
    print(f"  输出图像形状: {image_auto_period.shape}")
    print(f"  检测到的周期（约）: {image_auto_period.shape[1]}")
    
    # 4. 图像转回时序数据 - Image to Time Series
    print("\n4. Image to Time Series Conversion - 图像转回时序数据")
    print("-" * 70)
    
    # 单维图像转回数据
    recovered_1d = processor.from_image(
        image_1d, 
        norm_min=np.min(data_1d), 
        norm_max=np.max(data_1d)
    )
    
    print(f"单维图像转回数据:")
    print(f"  原始数据形状: {data_1d.shape}")
    print(f"  恢复数据形状: {recovered_1d.shape}")
    print(f"  重建误差: {np.mean(np.abs(data_1d - recovered_1d)):.6f}")
    
    # 多维图像转回数据
    recovered_2d = processor.from_image(
        image_2d,
        norm_min=np.min(data_2d),
        norm_max=np.max(data_2d)
    )
    
    print(f"\n多维图像转回数据:")
    print(f"  原始数据形状: {data_2d.shape}")
    print(f"  恢复数据形状: {recovered_2d.shape}")
    print(f"  重建误差: {np.mean(np.abs(data_2d - recovered_2d)):.6f}")
    
    # 5. 完整流程示例 - Complete Workflow
    print("\n5. Complete Workflow Example - 完整流程示例")
    print("-" * 70)
    
    # 生成模拟的传感器数据（3个通道：温度、压力、湿度）
    t = np.arange(120)
    temperature = 20 + 5 * np.sin(2 * np.pi * t / 24) + np.random.randn(120) * 0.5
    pressure = 1000 + 10 * np.sin(2 * np.pi * t / 24 + np.pi/4) + np.random.randn(120) * 1
    humidity = 60 + 15 * np.sin(2 * np.pi * t / 24 + np.pi/2) + np.random.randn(120) * 2
    
    sensor_data = np.column_stack([temperature, pressure, humidity])
    
    print(f"传感器数据形状: {sensor_data.shape}")
    print(f"特征: [温度, 压力, 湿度]")
    
    # 步骤1: 滤波去噪
    filtered_data = processor.filter(sensor_data, method='savgol', window_size=11, polyorder=2)
    print(f"\n步骤1 - 滤波去噪:")
    print(f"  温度噪声降低: {np.std(temperature):.4f} -> {np.std(filtered_data[:, 0]):.4f}")
    
    # 步骤2: 转换为图像（每24小时一行，3个通道）
    image = processor.to_image(filtered_data, period=24)
    print(f"\n步骤2 - 转换为图像:")
    print(f"  图像形状: {image.shape} (5天 × 24小时 × 3通道)")
    print(f"  可用于图像处理算法或CNN模型")
    
    # 步骤3: 分解第一个通道（温度）
    temp_components = processor.decompose(filtered_data[:, 0], method='additive', period=24)
    print(f"\n步骤3 - 温度数据分解:")
    print(f"  趋势标准差: {np.std(temp_components['trend']):.4f}")
    print(f"  季节标准差: {np.std(temp_components['seasonal']):.4f}")
    print(f"  残差标准差: {np.std(temp_components['residual']):.4f}")
    
    print("\n" + "=" * 70)
    print("Data Processing Example Completed! - 数据处理示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
