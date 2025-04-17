import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares

# --- Baseline: Matrix Factorization ---
def matrix_factorization(train_data, n_factors=50, n_epochs=20):
    # Create user-item matrix
    user_ids = train_data['user_id'].astype('category').cat.codes
    item_ids = train_data['business_id'].astype('category').cat.codes
    ratings = train_data['stars']
    sparse_matrix = csr_matrix((ratings, (user_ids, item_ids)))
    
    model = AlternatingLeastSquares(factors=n_factors, iterations=n_epochs, regularization=0.1)
    model.fit(sparse_matrix)
    
    def predict_mf(user_id, item_id, user_map, item_map):
        u_idx = user_map.get(user_id, -1)
        i_idx = item_map.get(item_id, -1)
        if u_idx == -1 or i_idx == -1:
            return 3.0  # Default prediction
        return np.dot(model.user_factors[u_idx], model.item_factors[i_idx])
    
    return predict_mf, dict(zip(train_data['user_id'], user_ids)), dict(zip(train_data['business_id'], item_ids))

# --- Baseline: Item-Based Collaborative Filtering ---
def item_based_cf(train_data):
    # Pivot to item-user matrix
    item_user_matrix = train_data.pivot(index='business_id', columns='user_id', values='stars').fillna(0)
    item_similarities = cosine_similarity(item_user_matrix)
    item_similarities = pd.DataFrame(item_similarities, index=item_user_matrix.index, columns=item_user_matrix.index)
    
    def predict_icf(user_id, item_id, train_data, similarities, k=10):
        user_ratings = train_data[train_data['user_id'] == user_id][['business_id', 'stars']]
        if item_id not in similarities.index:
            return 3.0
        sim_scores = similarities.loc[item_id]
        rated_items = user_ratings['business_id'].values
        valid_sims = sim_scores[rated_items]
        top_k = valid_sims.nlargest(k)
        if top_k.sum() == 0:
            return 3.0
        weighted_ratings = sum(user_ratings[user_ratings['business_id'].isin(top_k.index)]['stars'] * top_k) / top_k.sum()
        return weighted_ratings
    
    return predict_icf, item_similarities