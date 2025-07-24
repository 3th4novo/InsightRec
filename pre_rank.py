import pandas as pd

class HybridRanker:
    """
    混合排序模块：支持加法与乘法两种融合策略
    Attributes:
        alpha (float): CTR 分数权重
        beta (float): 召回分数权重
        mode (str): 'add' or 'mul'
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, mode: str = 'add'):
        assert mode in ('add', 'mul'), "mode must be 'add' or 'mul'"
        self.alpha = alpha
        self.beta = beta
        self.mode = mode

    def blend(self,
              recall_scores: pd.Series,
              ctr_scores: pd.Series) -> pd.Series:
        """
        对齐索引后，进行加权融合。

        Args:
            recall_scores: pd.Series, index 为 item_id，值为召回得分
            ctr_scores:    pd.Series, index 为 item_id，值为 CTR 预估
        Returns:
            pd.Series 排序后的混合得分（降序）
        """
        # 对齐所有 item
        df = pd.DataFrame({
            'recall': recall_scores,
            'ctr': ctr_scores
        }).fillna(0)

        if self.mode == 'add':
            df['score'] = self.alpha * df['ctr'] + self.beta * df['recall']
        else:
            # 乘法融合，对得分取指数后相乘
            df['score'] = (df['ctr'] ** self.alpha) * (df['recall'] ** self.beta)

        # 返回降序排列的 item_id -> score
        return df['score'].sort_values(ascending=False)

if __name__ == '__main__':
    # 示例数据
    recall = pd.Series({1: 0.8, 2: 0.5, 3: 0.3})
    ctr    = pd.Series({1: 0.2, 2: 0.4, 3: 0.9})

    # 加法混排
    ranker_add = HybridRanker(alpha=1.0, beta=1.0, mode='add')
    print("Additive blend:", ranker_add.blend(recall, ctr))

    # 乘法混排
    ranker_mul = HybridRanker(alpha=1.0, beta=1.0, mode='mul')
    print("Multiplicative blend:", ranker_mul.blend(recall, ctr))
