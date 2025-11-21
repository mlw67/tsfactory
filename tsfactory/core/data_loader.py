"""
Data Loader Module for Time Series Data

支持加载和处理单维度和多维度时序数据
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


class TimeSeriesLoader:
    """
    时序数据加载器
    
    支持加载和处理单维度和多维度时序数据，提供通用性的数据接口。
    """
    
    def __init__(self):
        self.data = None
        self.timestamps = None
        self.feature_names = None
        self.n_features = 0
        self.n_samples = 0
    
    def load_from_array(
        self, 
        data: np.ndarray, 
        timestamps: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> 'TimeSeriesLoader':
        """
        从numpy数组加载时序数据
        
        Parameters:
        -----------
        data : np.ndarray
            时序数据，shape可以是(n_samples,)或(n_samples, n_features)
        timestamps : np.ndarray, optional
            时间戳数组
        feature_names : list, optional
            特征名称列表
            
        Returns:
        --------
        self : TimeSeriesLoader
        """
        # 确保数据是2D的
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.data = data
        self.n_samples, self.n_features = data.shape
        
        if timestamps is None:
            self.timestamps = np.arange(self.n_samples)
        else:
            self.timestamps = timestamps
        
        if feature_names is None:
            if self.n_features == 1:
                self.feature_names = ['value']
            else:
                self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        else:
            self.feature_names = feature_names
        
        return self
    
    def load_from_dataframe(
        self, 
        df: pd.DataFrame, 
        timestamp_col: Optional[str] = None
    ) -> 'TimeSeriesLoader':
        """
        从pandas DataFrame加载时序数据
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含时序数据的DataFrame
        timestamp_col : str, optional
            时间戳列名，如果为None则使用索引
            
        Returns:
        --------
        self : TimeSeriesLoader
        """
        if timestamp_col is not None:
            self.timestamps = df[timestamp_col].values
            data_df = df.drop(columns=[timestamp_col])
        else:
            self.timestamps = df.index.values
            data_df = df
        
        self.data = data_df.values
        self.feature_names = list(data_df.columns)
        self.n_samples, self.n_features = self.data.shape
        
        return self
    
    def load_from_csv(
        self, 
        filepath: str, 
        timestamp_col: Optional[str] = None,
        **kwargs
    ) -> 'TimeSeriesLoader':
        """
        从CSV文件加载时序数据
        
        Parameters:
        -----------
        filepath : str
            CSV文件路径
        timestamp_col : str, optional
            时间戳列名
        **kwargs : dict
            传递给pd.read_csv的额外参数
            
        Returns:
        --------
        self : TimeSeriesLoader
        """
        df = pd.read_csv(filepath, **kwargs)
        return self.load_from_dataframe(df, timestamp_col)
    
    def get_data(self) -> np.ndarray:
        """获取数据数组"""
        return self.data
    
    def get_timestamps(self) -> np.ndarray:
        """获取时间戳数组"""
        return self.timestamps
    
    def get_feature_names(self) -> list:
        """获取特征名称列表"""
        return self.feature_names
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        转换为pandas DataFrame
        
        Returns:
        --------
        df : pd.DataFrame
            包含时序数据的DataFrame
        """
        df = pd.DataFrame(self.data, columns=self.feature_names)
        df.index = self.timestamps
        return df
    
    def split_train_test(
        self, 
        test_size: Union[int, float] = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割训练集和测试集
        
        Parameters:
        -----------
        test_size : int or float
            测试集大小。如果是float，表示比例；如果是int，表示样本数
            
        Returns:
        --------
        train_data : np.ndarray
            训练集数据
        test_data : np.ndarray
            测试集数据
        """
        if isinstance(test_size, float):
            split_idx = int(self.n_samples * (1 - test_size))
        else:
            split_idx = self.n_samples - test_size
        
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        
        return train_data, test_data
