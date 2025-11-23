"""
Tests for Deep Learning Models

深度学习模型测试
"""

import numpy as np
import pytest
import tempfile
import os
from tsfactory import DLinearLoader, TransformerLoader


class TestDLinearLoader:
    """测试DLinear模型加载器"""
    
    def test_initialization(self):
        """测试初始化"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        
        assert loader.seq_len == 96
        assert loader.pred_len == 24
        assert loader.enc_in == 1
        assert not loader.is_loaded
        assert loader.seasonal_weights is None
        assert loader.trend_weights is None
    
    def test_initialization_with_params(self):
        """测试带参数的初始化"""
        loader = DLinearLoader(
            seq_len=192,
            pred_len=48,
            enc_in=7,
            individual=True,
            kernel_size=51
        )
        
        assert loader.seq_len == 192
        assert loader.pred_len == 48
        assert loader.enc_in == 7
        assert loader.individual == True
        assert loader.kernel_size == 51
    
    def test_load_from_dict(self):
        """测试从字典加载"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        
        # Create dummy weights
        state_dict = {
            'seasonal_Linear.weight': np.random.randn(24, 96),
            'seasonal_Linear.bias': np.random.randn(24),
            'trend_Linear.weight': np.random.randn(24, 96),
            'trend_Linear.bias': np.random.randn(24)
        }
        
        loader.load_from_dict(state_dict)
        
        assert loader.is_loaded
        assert loader.seasonal_weights is not None
        assert loader.trend_weights is not None
        assert loader.seasonal_weights['weight'].shape == (24, 96)
        assert loader.trend_weights['weight'].shape == (24, 96)
    
    def test_load_from_dict_partial(self):
        """测试从字典加载部分权重"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        
        # Only seasonal weights
        state_dict = {
            'seasonal_Linear.weight': np.random.randn(24, 96),
            'seasonal_Linear.bias': np.random.randn(24),
        }
        
        loader.load_from_dict(state_dict)
        
        assert loader.is_loaded
        assert loader.seasonal_weights is not None
        assert loader.trend_weights is None
    
    def test_moving_average(self):
        """测试移动平均"""
        loader = DLinearLoader(seq_len=100, pred_len=10, kernel_size=5)
        
        # Create a simple trend
        x = np.arange(100).reshape(1, 100, 1).astype(float)
        
        trend = loader._moving_average(x, kernel_size=5)
        
        assert trend.shape == x.shape
        # Moving average should smooth the data
        assert np.std(trend) <= np.std(x)
    
    def test_series_decomp(self):
        """测试时序分解"""
        loader = DLinearLoader(seq_len=100, pred_len=10, kernel_size=5)
        
        # Create data with trend and seasonality
        t = np.arange(100)
        trend_component = 0.5 * t
        seasonal_component = 10 * np.sin(2 * np.pi * t / 12)
        x = (trend_component + seasonal_component).reshape(1, 100, 1)
        
        seasonal, trend = loader._series_decomp(x)
        
        assert seasonal.shape == x.shape
        assert trend.shape == x.shape
        # Seasonal + trend should equal original
        np.testing.assert_array_almost_equal(seasonal + trend, x, decimal=10)
    
    def test_predict_1d(self):
        """测试一维数据预测"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        
        # Load dummy weights
        state_dict = {
            'seasonal_Linear.weight': np.random.randn(24, 96),
            'seasonal_Linear.bias': np.random.randn(24),
            'trend_Linear.weight': np.random.randn(24, 96),
            'trend_Linear.bias': np.random.randn(24)
        }
        loader.load_from_dict(state_dict)
        
        # Create input data
        x = np.random.randn(96)
        
        # Make predictions
        predictions = loader.predict(x)
        
        assert predictions.shape == (24,)
    
    def test_predict_2d(self):
        """测试二维数据预测"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=3)
        
        # Load dummy weights  
        state_dict = {
            'seasonal_Linear.weight': np.random.randn(72, 288),  # 24*3, 96*3
            'seasonal_Linear.bias': np.random.randn(72),
            'trend_Linear.weight': np.random.randn(72, 288),
            'trend_Linear.bias': np.random.randn(72)
        }
        loader.load_from_dict(state_dict)
        
        # Create input data (seq_len, n_features)
        x = np.random.randn(96, 3)
        
        # Make predictions
        predictions = loader.predict(x)
        
        assert predictions.shape == (24, 3)
    
    def test_predict_batch(self):
        """测试批量预测"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        
        # Load dummy weights
        state_dict = {
            'seasonal_Linear.weight': np.random.randn(24, 96),
            'seasonal_Linear.bias': np.random.randn(24),
            'trend_Linear.weight': np.random.randn(24, 96),
            'trend_Linear.bias': np.random.randn(24)
        }
        loader.load_from_dict(state_dict)
        
        # Create batch input data
        x = np.random.randn(4, 96, 1)
        
        # Make predictions
        predictions = loader.predict(x)
        
        assert predictions.shape == (4, 24, 1)
    
    def test_predict_without_loading(self):
        """测试未加载模型时预测"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        x = np.random.randn(96)
        
        with pytest.raises(ValueError, match="Model must be loaded"):
            loader.predict(x)
    
    def test_predict_wrong_seq_len(self):
        """测试错误的序列长度"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        
        state_dict = {
            'seasonal_Linear.weight': np.random.randn(24, 96),
            'seasonal_Linear.bias': np.random.randn(24),
        }
        loader.load_from_dict(state_dict)
        
        # Wrong sequence length
        x = np.random.randn(48)
        
        with pytest.raises(ValueError, match="Input sequence length"):
            loader.predict(x)
    
    def test_load_from_npz_file(self):
        """测试从npz文件加载"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save weights to file
            np.savez(
                temp_path,
                **{
                    'seasonal_Linear.weight': np.random.randn(24, 96),
                    'seasonal_Linear.bias': np.random.randn(24),
                    'trend_Linear.weight': np.random.randn(24, 96),
                    'trend_Linear.bias': np.random.randn(24)
                }
            )
            
            # Load from file
            loader.load_from_file(temp_path)
            
            assert loader.is_loaded
            assert loader.seasonal_weights is not None
            
        finally:
            os.unlink(temp_path)
    
    def test_load_from_npy_file(self):
        """测试从npy文件加载"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=1)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save weights as dict in npy
            state_dict = {
                'seasonal_Linear.weight': np.random.randn(24, 96),
                'seasonal_Linear.bias': np.random.randn(24),
            }
            np.save(temp_path, state_dict)
            
            # Load from file
            loader.load_from_file(temp_path)
            
            assert loader.is_loaded
            
        finally:
            os.unlink(temp_path)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        loader = DLinearLoader(seq_len=96, pred_len=24, enc_in=7, kernel_size=25)
        
        info = loader.get_model_info()
        
        assert info['model_type'] == 'DLinear'
        assert info['seq_len'] == 96
        assert info['pred_len'] == 24
        assert info['enc_in'] == 7
        assert info['kernel_size'] == 25
        assert not info['is_loaded']
        assert not info['has_seasonal_weights']
        assert not info['has_trend_weights']
        
        # Load weights
        state_dict = {
            'seasonal_Linear.weight': np.random.randn(24, 96),
        }
        loader.load_from_dict(state_dict)
        
        info = loader.get_model_info()
        assert info['is_loaded']
        assert info['has_seasonal_weights']


class TestTransformerLoader:
    """测试Transformer模型加载器"""
    
    def test_initialization(self):
        """测试初始化"""
        loader = TransformerLoader(seq_len=96, pred_len=24)
        
        assert loader.seq_len == 96
        assert loader.pred_len == 24
        assert loader.d_model == 512
        assert loader.n_heads == 8
        assert not loader.is_loaded
    
    def test_initialization_with_params(self):
        """测试带参数的初始化"""
        loader = TransformerLoader(
            seq_len=192,
            pred_len=48,
            d_model=256,
            n_heads=4,
            e_layers=3,
            d_layers=2,
            d_ff=1024,
            enc_in=7,
            dec_in=7,
            c_out=7,
            dropout=0.2
        )
        
        assert loader.seq_len == 192
        assert loader.pred_len == 48
        assert loader.d_model == 256
        assert loader.n_heads == 4
        assert loader.e_layers == 3
        assert loader.d_layers == 2
        assert loader.d_ff == 1024
        assert loader.dropout == 0.2
    
    def test_load_from_dict(self):
        """测试从字典加载"""
        loader = TransformerLoader(seq_len=96, pred_len=24)
        
        # Create dummy weights with encoder, decoder, projection
        state_dict = {
            'encoder.layer0.weight': np.random.randn(512, 512),
            'encoder.layer0.bias': np.random.randn(512),
            'decoder.layer0.weight': np.random.randn(512, 512),
            'decoder.layer0.bias': np.random.randn(512),
            'projection.weight': np.random.randn(24, 96*512),
            'projection.bias': np.random.randn(24)
        }
        
        loader.load_from_dict(state_dict)
        
        assert loader.is_loaded
        assert loader.encoder_weights is not None
        assert loader.decoder_weights is not None
        assert loader.output_projection is not None
    
    def test_load_from_dict_partial(self):
        """测试从字典加载部分权重"""
        loader = TransformerLoader(seq_len=96, pred_len=24)
        
        # Only encoder weights
        state_dict = {
            'encoder.layer0.weight': np.random.randn(512, 512),
            'encoder.layer0.bias': np.random.randn(512),
        }
        
        loader.load_from_dict(state_dict)
        
        assert loader.is_loaded
        assert loader.encoder_weights is not None
        assert loader.decoder_weights is None
        assert loader.output_projection is None
    
    def test_positional_encoding(self):
        """测试位置编码"""
        loader = TransformerLoader(seq_len=100, pred_len=10, d_model=512)
        
        pe = loader._positional_encoding(length=100, d_model=512)
        
        assert pe.shape == (100, 512)
        # Check that encoding has correct periodic properties
        assert not np.allclose(pe[0], pe[1])
        
        # Verify first position encoding
        position = 0
        div_term = np.exp(np.arange(0, 512, 2) * -(np.log(10000.0) / 512))
        expected_sin = np.sin(position * div_term)
        expected_cos = np.cos(position * div_term)
        
        # For position 0, sin(0) = 0 and cos(0) = 1
        assert np.allclose(pe[0, 0::2], expected_sin)
        assert np.allclose(pe[0, 1::2], expected_cos)
        
        # Test with odd d_model
        pe_odd = loader._positional_encoding(length=100, d_model=513)
        assert pe_odd.shape == (100, 513)
    
    def test_predict_with_projection(self):
        """测试使用投影层预测"""
        loader = TransformerLoader(seq_len=96, pred_len=24, enc_in=1, c_out=1)
        
        # Create projection weights that match dimensions
        state_dict = {
            'projection.weight': np.random.randn(24, 96),
            'projection.bias': np.random.randn(24)
        }
        loader.load_from_dict(state_dict)
        
        # Create input data
        x = np.random.randn(96)
        
        # Make predictions (will use simplified projection)
        with pytest.warns(UserWarning, match="Using simplified prediction"):
            predictions = loader.predict(x, use_simple_projection=True)
        
        assert predictions.shape == (24,)
    
    def test_predict_fallback(self):
        """测试后备预测方法"""
        loader = TransformerLoader(seq_len=96, pred_len=24, enc_in=1)
        
        # Load empty state dict
        loader.load_from_dict({})
        
        x = np.random.randn(96)
        
        # Should use fallback (last value repetition)
        with pytest.warns(UserWarning, match="Could not perform full prediction"):
            predictions = loader.predict(x)
        
        assert predictions.shape == (24,)
        # Fallback repeats last value
        assert np.allclose(predictions, x[-1])
    
    def test_predict_batch(self):
        """测试批量预测"""
        loader = TransformerLoader(seq_len=96, pred_len=24, enc_in=1, c_out=1)
        
        state_dict = {
            'projection.weight': np.random.randn(24, 96),
            'projection.bias': np.random.randn(24)
        }
        loader.load_from_dict(state_dict)
        
        # Create batch input
        x = np.random.randn(4, 96, 1)
        
        with pytest.warns(UserWarning):
            predictions = loader.predict(x)
        
        assert predictions.shape == (4, 24, 1)
    
    def test_predict_without_loading(self):
        """测试未加载模型时预测"""
        loader = TransformerLoader(seq_len=96, pred_len=24)
        x = np.random.randn(96)
        
        with pytest.raises(ValueError, match="Model must be loaded"):
            loader.predict(x)
    
    def test_predict_wrong_seq_len(self):
        """测试错误的序列长度"""
        loader = TransformerLoader(seq_len=96, pred_len=24)
        loader.load_from_dict({})
        
        # Wrong sequence length
        x = np.random.randn(48)
        
        with pytest.raises(ValueError, match="Input sequence length"):
            loader.predict(x)
    
    def test_load_from_npz_file(self):
        """测试从npz文件加载"""
        loader = TransformerLoader(seq_len=96, pred_len=24)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
        
        try:
            np.savez(
                temp_path,
                **{
                    'encoder.weight': np.random.randn(512, 512),
                    'decoder.weight': np.random.randn(512, 512),
                }
            )
            
            loader.load_from_file(temp_path)
            
            assert loader.is_loaded
            assert loader.encoder_weights is not None
            
        finally:
            os.unlink(temp_path)
    
    def test_load_from_npy_file(self):
        """测试从npy文件加载"""
        loader = TransformerLoader(seq_len=96, pred_len=24)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            state_dict = {
                'encoder.weight': np.random.randn(512, 512),
            }
            np.save(temp_path, state_dict)
            
            loader.load_from_file(temp_path)
            
            assert loader.is_loaded
            
        finally:
            os.unlink(temp_path)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        loader = TransformerLoader(
            seq_len=96,
            pred_len=24,
            d_model=256,
            n_heads=4,
            e_layers=2
        )
        
        info = loader.get_model_info()
        
        assert info['model_type'] == 'Transformer'
        assert info['seq_len'] == 96
        assert info['pred_len'] == 24
        assert info['d_model'] == 256
        assert info['n_heads'] == 4
        assert info['e_layers'] == 2
        assert not info['is_loaded']
        assert not info['has_encoder_weights']
        assert not info['has_decoder_weights']
        assert not info['has_output_projection']
        
        # Load weights
        state_dict = {
            'encoder.weight': np.random.randn(256, 256),
            'projection.weight': np.random.randn(24, 96)
        }
        loader.load_from_dict(state_dict)
        
        info = loader.get_model_info()
        assert info['is_loaded']
        assert info['has_encoder_weights']
        assert info['has_output_projection']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
