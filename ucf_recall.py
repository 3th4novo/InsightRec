import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class UserCFRecall:
    """
    User-based Collaborative Filtering Recall Module

    Attributes:
        ratings: pd.DataFrame with columns [user_id, item_id, rating]
        user_item_matrix: pd.DataFrame user-item sparse matrix
        user_similarity: pd.DataFrame user-user similarity matrix
    """
    def __init__(self, ratings: pd.DataFrame, user_key: str, item_key: str, rating_key: str = None):
        self.user_key = user_key
        self.item_key = item_key
        self.rating_key = rating_key
        # Build user-item matrix
        if rating_key:
            self.user_item_matrix = ratings.pivot_table(
                index=user_key,
                columns=item_key,
                values=rating_key,
                fill_value=0
            )
        else:
            # binary implicit feedback
            ratings['interaction'] = 1
            self.user_item_matrix = ratings.pivot_table(
                index=user_key,
                columns=item_key,
                values='interaction',
                fill_value=0
            )
        # Compute similarity
        self.user_similarity = self._compute_user_similarity()

    def _compute_user_similarity(self) -> pd.DataFrame:
        """Compute cosine similarity between users."""
        # user-item matrix as numpy
        mat = self.user_item_matrix.values
        sim = cosine_similarity(mat)
        sim_df = pd.DataFrame(
            sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        return sim_df

    def recall(self, user_id, top_n: int = 300, k_sim: int = 10) -> list:
        """
        Recall candidate items for a given user.

        Args:
            user_id: target user
            top_n: number of items to recall
            k_sim: number of most similar users to consider

        Returns:
            List of item_ids (length <= top_n)
        """
        # 1. Find top-k similar users
        if user_id not in self.user_similarity.index:
            return []
        sim_scores = self.user_similarity.loc[user_id]
        top_users = sim_scores.drop(user_id).nlargest(k_sim)

        # 2. Aggregate their interactions
        neighbors = top_users.index.tolist()
        neighbor_weights = top_users.values

        # Get the items neighbors have interacted with
        neighbor_matrix = self.user_item_matrix.loc[neighbors]
        # Weighted sum of neighbor interactions
        weighted_sum = neighbor_matrix.T.dot(neighbor_weights)
        weighted_sum = pd.Series(weighted_sum, index=self.user_item_matrix.columns)

        # 3. Filter out items the user has already seen
        user_seen = set(self.user_item_matrix.loc[user_id].replace(0, np.nan).dropna().index)
        candidates = weighted_sum.drop(labels=list(user_seen), errors='ignore')

        # 4. Return top-n items
        recommended_items = candidates.nlargest(top_n).index.tolist()
        return recommended_items

if __name__ == "__main__":
    # Example usage with MovieLens 1M
    # Load ratings: columns userId, movieId, rating
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    # Initialize module
    ucf = UserCFRecall(ratings, user_key='userId', item_key='movieId', rating_key='rating')
    # Recall for user 123
    recs = ucf.recall(user_id=123, top_n=300, k_sim=20)
    print(f"Top-{len(recs)} recommendations for user 123:", recs[:20])
