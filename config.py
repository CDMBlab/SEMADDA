# config.py
import argparse

class Config:
    """配置类"""
    
    # Transformer配置
    USE_TRANSFORMER = True
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 8
    TRANSFORMER_DIM = 512
    
    # 训练配置
    LEARNING_RATE = 2e-3
    DROPOUT_RATE = 0.4
    EPOCHS = 12
    BATCH_SIZE = 8
    
    # 模型配置
    HIDDEN_DIM = 128
    LAYERS = 3
    
    # 数据配置
    DATA_NAME = 'f'
    TEST_RATIO = 10
    MAX_TRAIN_NUM = 10000
    HOP = 2
    SIM_WEIGHT = 2
    
    # 训练策略
    USE_CONTRASTIVE = True
    USE_HIERARCHY = True
    MODEL_TYPE = 'hierarchical'
    GRAD_ACCUMULATION = 4
    WARMUP_RATIO = 0.1
    CONTRASTIVE_WEIGHT = 0.1
    
    @classmethod
    def create_args(cls):
        """创建命令行参数"""
        parser = argparse.ArgumentParser()
        
        # 基础配置
        parser.add_argument('--data-name', default=cls.DATA_NAME)
        parser.add_argument('--lr', type=float, default=cls.LEARNING_RATE)
        parser.add_argument('--dropout-rate', type=float, default=cls.DROPOUT_RATE)
        parser.add_argument('--epoch', type=int, default=cls.EPOCHS)
        parser.add_argument('--hidden', type=int, default=cls.HIDDEN_DIM)
        parser.add_argument('--layer', type=int, default=cls.LAYERS)
        parser.add_argument('--test-ratio', type=int, default=cls.TEST_RATIO)
        parser.add_argument('--max-train-num', type=int, default=cls.MAX_TRAIN_NUM)
        parser.add_argument('--hop', default=cls.HOP)
        parser.add_argument('--simw', type=int, default=cls.SIM_WEIGHT)
        
        # Transformer配置
        parser.add_argument('--use-transformer', action='store_true', default=cls.USE_TRANSFORMER)
        parser.add_argument('--transformer-layers', type=int, default=cls.TRANSFORMER_LAYERS)
        parser.add_argument('--transformer-heads', type=int, default=cls.TRANSFORMER_HEADS)
        parser.add_argument('--transformer-dim', type=int, default=cls.TRANSFORMER_DIM)
        
        # 训练策略
        parser.add_argument('--use-contrastive', action='store_true', default=cls.USE_CONTRASTIVE)
        parser.add_argument('--use-hierarchy', action='store_true', default=cls.USE_HIERARCHY)
        parser.add_argument('--model-type', type=str, default=cls.MODEL_TYPE)
        parser.add_argument('--grad-accumulation', type=int, default=cls.GRAD_ACCUMULATION)
        parser.add_argument('--warmup-ratio', type=float, default=cls.WARMUP_RATIO)
        parser.add_argument('--contrastive-weight', type=float, default=cls.CONTRASTIVE_WEIGHT)
        
        # 其他参数
        parser.add_argument('--no-cuda', action='store_true', default=True)
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--pro', type=int, default=1)
        parser.add_argument('--max-nodes-per-hop', default=100)
        parser.add_argument('--batch-size-graph', type=int, default=100)
        
        args = parser.parse_args([])  # 空列表表示使用默认值
        return args