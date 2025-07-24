import pandas as pd
import numpy as np
import time
from ucf_recall import UserCFRecall
from pre_rank import HybridRanker


def main():
    total_start = time.perf_counter()

    # 1. 加载评分数据
    start = time.perf_counter()
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    load_time = time.perf_counter() - start
    print(f"Data load time: {load_time:.3f}s")

    # 2. 初始化召回 & 混排模块
    start = time.perf_counter()
    ucf = UserCFRecall(ratings, user_key='userId', item_key='movieId', rating_key='rating')
    ranker_add = HybridRanker(alpha=1.0, beta=1.0, mode='add')
    ranker_mul = HybridRanker(alpha=1.0, beta=1.0, mode='mul')
    init_time = time.perf_counter() - start
    print(f"Module initialization time: {init_time:.3f}s")

    user_id = 123

    # 3. 执行 UCF 召回 (Top-300)
    start = time.perf_counter()
    recall_items = ucf.recall(user_id=user_id, top_n=300, k_sim=20)
    recall_time = time.perf_counter() - start
    print(f"Recall time: {recall_time:.3f}s (returned {len(recall_items)} items)")

    # 4. 重新计算召回分数
    start = time.perf_counter()
    sim_scores = ucf.user_similarity.loc[user_id]
    top_users = sim_scores.drop(user_id).nlargest(20)
    neighbors, weights = top_users.index.tolist(), top_users.values
    neighbor_matrix = ucf.user_item_matrix.loc[neighbors]
    weighted_sum = neighbor_matrix.T.dot(weights)
    seen = set(ucf.user_item_matrix.loc[user_id].replace(0, np.nan).dropna().index)
    recall_scores = weighted_sum.drop(labels=list(seen), errors='ignore').nlargest(300)
    score_time = time.perf_counter() - start
    print(f"Recall score computation time: {score_time:.3f}s")

    # 5. 获取 CTR 预估分数（此处用随机数模拟）
    start = time.perf_counter()
    np.random.seed(42)
    ctr_scores = pd.Series(index=recall_scores.index,
                           data=np.random.rand(len(recall_scores)))
    ctr_time = time.perf_counter() - start
    print(f"CTR prediction time (simulated): {ctr_time:.3f}s")

    # 6. 混排 & 输出 Top-20
    start = time.perf_counter()
    final_add = ranker_add.blend(recall_scores, ctr_scores).head(20)
    add_time = time.perf_counter() - start
    print(f"Additive blend time: {add_time:.3f}s")

    start = time.perf_counter()
    final_mul = ranker_mul.blend(recall_scores, ctr_scores).head(20)
    mul_time = time.perf_counter() - start
    print(f"Multiplicative blend time: {mul_time:.3f}s")

    total_time = time.perf_counter() - total_start
    print(f"Total runtime: {total_time:.3f}s")

    # 打印结果
    print("\n=== Additive Mixed Ranking Top-20 ===")
    print(final_add)
    print("\n=== Multiplicative Mixed Ranking Top-20 ===")
    print(final_mul)


if __name__ == '__main__':
    main()
