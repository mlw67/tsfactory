"""
Deep Learning Data Loaders

深度学习数据加载器，用于加载预训练的深度学习模型权重
"""

import numpy as np
from typing import Dict, Any


class DLinearDataLoader:
    """
    DLinear模型数据加载器
    
    Data loader for DLinear models. Loads pre-trained model weights
    from various file formats.
    """
    
    @staticmethod
    def load_from_dict(state_dict: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        从字典加载模型参数
        Load model parameters from dictionary
        
        Parameters:
        -----------
        state_dict : dict
            包含模型权重的字典 / Dictionary containing model weights
            Expected keys: 'seasonal_Linear.weight', 'seasonal_Linear.bias',
                          'trend_Linear.weight', 'trend_Linear.bias'
                          
        Returns:
        --------
        weights : dict
            包含 'seasonal' 和 'trend' 的权重字典
        """
        weights = {}
        
        # Load seasonal linear layer weights
        if 'seasonal_Linear.weight' in state_dict:
            weights['seasonal'] = {
                'weight': np.array(state_dict['seasonal_Linear.weight']),
                'bias': np.array(state_dict.get('seasonal_Linear.bias', 0))
            }
        
        # Load trend linear layer weights  
        if 'trend_Linear.weight' in state_dict:
            weights['trend'] = {
                'weight': np.array(state_dict['trend_Linear.weight']),
                'bias': np.array(state_dict.get('trend_Linear.bias', 0))
            }
        
        return weights
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        从文件加载模型
        Load model from file
        
        Parameters:
        -----------
        filepath : str
            模型文件路径 / Path to model file (.npy, .npz, or PyTorch .pth/.pt)
            
        Returns:
        --------
        weights : dict
            包含 'seasonal' 和 'trend' 的权重字典
        """
        try:
            if filepath.endswith('.npz'):
                # Load from numpy archive
                data = np.load(filepath)
                state_dict = {key: data[key] for key in data.files}
                return DLinearDataLoader.load_from_dict(state_dict)
            elif filepath.endswith('.npy'):
                # Load from single numpy array
                data = np.load(filepath, allow_pickle=True).item()
                return DLinearDataLoader.load_from_dict(data)
            elif filepath.endswith(('.pth', '.pt')):
                # Try to load PyTorch model
                try:
                    import torch
                    checkpoint = torch.load(filepath, map_location='cpu')
                    # Convert torch tensors to numpy
                    state_dict = {k: v.cpu().numpy() for k, v in checkpoint.items()}
                    return DLinearDataLoader.load_from_dict(state_dict)
                except ImportError:
                    raise ImportError("PyTorch is required to load .pth/.pt files. Install it with: pip install torch")
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {str(e)}")


class TransformerDataLoader:
    """
    Transformer模型数据加载器
    
    Data loader for Transformer models. Loads pre-trained model weights
    from various file formats.
    """
    
    @staticmethod
    def load_from_dict(state_dict: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        从字典加载模型参数
        Load model parameters from dictionary
        
        Parameters:
        -----------
        state_dict : dict
            包含模型权重的字典 / Dictionary containing model weights
            Expected keys include encoder, decoder, and projection layer weights
            
        Returns:
        --------
        weights : dict
            包含 'encoder', 'decoder', 'projection' 的权重字典
        """
        weights = {}
        
        # Organize weights by component
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k.lower()]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k.lower()]
        projection_keys = [k for k in state_dict.keys() if 'projection' in k.lower() or 'output' in k.lower()]
        
        if encoder_keys:
            weights['encoder'] = {k: np.array(state_dict[k]) for k in encoder_keys}
        
        if decoder_keys:
            weights['decoder'] = {k: np.array(state_dict[k]) for k in decoder_keys}
        
        if projection_keys:
            weights['projection'] = {k: np.array(state_dict[k]) for k in projection_keys}
        
        return weights
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        从文件加载模型
        Load model from file
        
        Parameters:
        -----------
        filepath : str
            模型文件路径 / Path to model file (.npy, .npz, or PyTorch .pth/.pt)
            
        Returns:
        --------
        weights : dict
            包含 'encoder', 'decoder', 'projection' 的权重字典
        """
        try:
            if filepath.endswith('.npz'):
                # Load from numpy archive
                data = np.load(filepath)
                state_dict = {key: data[key] for key in data.files}
                return TransformerDataLoader.load_from_dict(state_dict)
            elif filepath.endswith('.npy'):
                # Load from single numpy array
                data = np.load(filepath, allow_pickle=True).item()
                return TransformerDataLoader.load_from_dict(data)
            elif filepath.endswith(('.pth', '.pt')):
                # Try to load PyTorch model
                try:
                    import torch
                    checkpoint = torch.load(filepath, map_location='cpu')
                    # Convert torch tensors to numpy
                    state_dict = {k: v.cpu().numpy() for k, v in checkpoint.items()}
                    return TransformerDataLoader.load_from_dict(state_dict)
                except ImportError:
                    raise ImportError("PyTorch is required to load .pth/.pt files. Install it with: pip install torch")
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {str(e)}")
