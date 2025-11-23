"""
Deep Learning Models Example: Using DLinear and Transformer for Time Series

深度学习模型示例：使用DLinear和Transformer进行时序预测
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# 导入深度学习模型加载器
from tsfactory import DLinearLoader, TransformerLoader


def generate_sample_data(n=200, add_trend=True, add_seasonality=True, add_noise=True):
    """
    生成示例时序数据
    """
    t = np.arange(n)
    data = np.zeros(n)
    
    # 趋势
    if add_trend:
        data += 0.05 * t
    
    # 季节性
    if add_seasonality:
        data += 10 * np.sin(2 * np.pi * t / 12)
    
    # 噪声
    if add_noise:
        data += np.random.randn(n) * 2
    
    return data


def main():
    print("=" * 70)
    print("Deep Learning Models Example - 深度学习模型示例")
    print("=" * 70)
    
    # 生成示例数据
    print("\n1. Generating Sample Data - 生成示例数据")
    print("-" * 70)
    
    np.random.seed(42)  # 设置随机种子以获得可重复的结果
    data = generate_sample_data(n=200)
    
    # 分割为训练集和测试集
    seq_len = 96
    pred_len = 24
    
    train_data = data[:seq_len]
    test_data = data[seq_len:seq_len+pred_len]
    
    print(f"Training data length: {len(train_data)}")
    print(f"Testing data length: {len(test_data)}")
    print(f"Sequence length (input): {seq_len}")
    print(f"Prediction length (output): {pred_len}")
    
    # =========================================================================
    # DLinear模型示例
    # =========================================================================
    print("\n2. DLinear Model Example - DLinear模型示例")
    print("-" * 70)
    
    # 初始化DLinear加载器
    dlinear = DLinearLoader(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=1,  # 单变量时序
        kernel_size=25  # 移动平均核大小
    )
    
    print("DLinear loader initialized:")
    print(f"  - Sequence length: {dlinear.seq_len}")
    print(f"  - Prediction length: {dlinear.pred_len}")
    print(f"  - Input features: {dlinear.enc_in}")
    print(f"  - Kernel size: {dlinear.kernel_size}")
    
    # 创建模拟的预训练权重
    # 在实际应用中，这些权重应该从训练好的PyTorch模型中加载
    print("\nLoading model weights...")
    np.random.seed(123)  # 固定随机种子
    state_dict = {
        'seasonal_Linear.weight': np.random.randn(pred_len, seq_len) * 0.01,
        'seasonal_Linear.bias': np.zeros(pred_len),
        'trend_Linear.weight': np.random.randn(pred_len, seq_len) * 0.01,
        'trend_Linear.bias': np.zeros(pred_len)
    }
    
    dlinear.load_from_dict(state_dict)
    print(f"Model loaded: {dlinear.is_loaded}")
    
    # 获取模型信息
    model_info = dlinear.get_model_info()
    print("\nModel information:")
    print(f"  - Model type: {model_info['model_type']}")
    print(f"  - Has seasonal weights: {model_info['has_seasonal_weights']}")
    print(f"  - Has trend weights: {model_info['has_trend_weights']}")
    
    # 进行预测
    print("\nMaking predictions with DLinear...")
    predictions = dlinear.predict(train_data)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5]}")
    
    # 保存原始预测以供后续比较
    original_predictions = predictions.copy()
    original_state_dict = {k: v.copy() for k, v in state_dict.items()}
    
    # 计算简单的评估指标
    # 注意：由于使用的是随机权重，预测精度会很低
    # 在实际应用中应使用真实训练的模型
    print("\nNote: Using random weights for demonstration.")
    print("In practice, load weights from a trained model for accurate predictions.")
    
    # =========================================================================
    # Transformer模型示例
    # =========================================================================
    print("\n3. Transformer Model Example - Transformer模型示例")
    print("-" * 70)
    
    # 初始化Transformer加载器
    transformer = TransformerLoader(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=512,  # 模型维度
        n_heads=8,    # 注意力头数
        e_layers=2,   # 编码器层数
        d_layers=1,   # 解码器层数
        d_ff=2048,    # 前馈网络维度
        enc_in=1,     # 编码器输入维度
        dec_in=1,     # 解码器输入维度
        c_out=1       # 输出维度
    )
    
    print("Transformer loader initialized:")
    print(f"  - Sequence length: {transformer.seq_len}")
    print(f"  - Prediction length: {transformer.pred_len}")
    print(f"  - Model dimension: {transformer.d_model}")
    print(f"  - Attention heads: {transformer.n_heads}")
    print(f"  - Encoder layers: {transformer.e_layers}")
    print(f"  - Decoder layers: {transformer.d_layers}")
    
    # 创建模拟的预训练权重
    print("\nLoading model weights...")
    state_dict = {
        'encoder.layer0.weight': np.random.randn(512, 512) * 0.01,
        'encoder.layer0.bias': np.zeros(512),
        'decoder.layer0.weight': np.random.randn(512, 512) * 0.01,
        'decoder.layer0.bias': np.zeros(512),
        'projection.weight': np.random.randn(pred_len, seq_len) * 0.01,
        'projection.bias': np.zeros(pred_len)
    }
    
    transformer.load_from_dict(state_dict)
    print(f"Model loaded: {transformer.is_loaded}")
    
    # 获取模型信息
    model_info = transformer.get_model_info()
    print("\nModel information:")
    print(f"  - Model type: {model_info['model_type']}")
    print(f"  - Has encoder weights: {model_info['has_encoder_weights']}")
    print(f"  - Has decoder weights: {model_info['has_decoder_weights']}")
    print(f"  - Has output projection: {model_info['has_output_projection']}")
    
    # 进行预测（使用简化的投影方法）
    print("\nMaking predictions with Transformer...")
    print("Note: Using simplified projection for demonstration.")
    predictions = transformer.predict(train_data, use_simple_projection=True)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5]}")
    
    # =========================================================================
    # 多维数据示例
    # =========================================================================
    print("\n4. Multi-dimensional Data Example - 多维数据示例")
    print("-" * 70)
    
    # 生成3维时序数据
    multi_data = np.column_stack([
        generate_sample_data(n=seq_len),
        generate_sample_data(n=seq_len),
        generate_sample_data(n=seq_len)
    ])
    
    print(f"Multi-dimensional data shape: {multi_data.shape}")
    print(f"Number of features: {multi_data.shape[1]}")
    
    # 初始化多维DLinear模型
    dlinear_multi = DLinearLoader(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=3,  # 3个特征
        kernel_size=25
    )
    
    # 加载权重（需要调整维度以匹配3个特征）
    state_dict_multi = {
        'seasonal_Linear.weight': np.random.randn(pred_len * 3, seq_len * 3) * 0.01,
        'seasonal_Linear.bias': np.zeros(pred_len * 3),
        'trend_Linear.weight': np.random.randn(pred_len * 3, seq_len * 3) * 0.01,
        'trend_Linear.bias': np.zeros(pred_len * 3)
    }
    
    dlinear_multi.load_from_dict(state_dict_multi)
    
    print("\nMaking predictions on multi-dimensional data...")
    predictions_multi = dlinear_multi.predict(multi_data)
    
    print(f"Multi-dimensional predictions shape: {predictions_multi.shape}")
    print(f"Predictions for first feature (first 3 values): {predictions_multi[:3, 0]}")
    
    # =========================================================================
    # 批量预测示例
    # =========================================================================
    print("\n5. Batch Prediction Example - 批量预测示例")
    print("-" * 70)
    
    # 创建批量数据（4个样本）
    batch_data = np.array([
        generate_sample_data(n=seq_len) for _ in range(4)
    ]).reshape(4, seq_len, 1)
    
    print(f"Batch data shape: {batch_data.shape}")
    
    # 使用之前的DLinear模型进行批量预测
    print("\nMaking batch predictions with DLinear...")
    batch_predictions = dlinear.predict(batch_data)
    
    print(f"Batch predictions shape: {batch_predictions.shape}")
    print(f"Predictions for first sample (first 3 values): {batch_predictions[0, :3, 0]}")
    
    # =========================================================================
    # 保存和加载模型示例
    # =========================================================================
    print("\n6. Save and Load Model Example - 保存和加载模型示例")
    print("-" * 70)
    
    import tempfile
    
    # 保存模型权重到文件
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name
    
    try:
        # 保存权重（使用原始保存的state_dict副本）
        np.savez(
            temp_path,
            **original_state_dict
        )
        print(f"Model weights saved to: {temp_path}")
        
        # 创建新的加载器并从文件加载
        dlinear_new = DLinearLoader(seq_len=seq_len, pred_len=pred_len, enc_in=1)
        dlinear_new.load_from_file(temp_path)
        
        print(f"Model loaded from file: {dlinear_new.is_loaded}")
        
        # 验证加载的模型可以正常预测（使用相同的输入数据）
        predictions_new = dlinear_new.predict(train_data)
        print(f"Predictions from loaded model shape: {predictions_new.shape}")
        
        # 验证预测结果是否一致（应该完全相同，因为使用相同的输入数据和权重）
        if np.allclose(original_predictions, predictions_new, rtol=1e-5):
            print("✓ Predictions match! Model saved and loaded successfully.")
        else:
            print("✗ Predictions differ. There may be an issue with loading.")
            print(f"  Max difference: {np.max(np.abs(original_predictions - predictions_new))}")
            
    finally:
        os.unlink(temp_path)
        print(f"Temporary file cleaned up.")
    
    print("\n" + "=" * 70)
    print("Deep Learning Example Completed! - 深度学习示例完成！")
    print("=" * 70)
    print("\nNote: This example uses randomly initialized weights for demonstration.")
    print("In production, you should:")
    print("  1. Train models using PyTorch or other deep learning frameworks")
    print("  2. Export trained model weights")
    print("  3. Load the weights using DLinearLoader or TransformerLoader")
    print("  4. Use for inference on new time series data")


if __name__ == "__main__":
    main()
