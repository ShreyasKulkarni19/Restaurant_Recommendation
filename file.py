import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
from itertools import product
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# --- Data Preprocessing ---
def load_and_preprocess_data(business_file, user_file, review_file, max_categories=50):
    # Load data
    business_df = pd.read_json(business_file, lines=True)
    user_df = pd.read_json(user_file, lines=True)
    review_df = pd.read_json(review_file, lines=True)

    # Filter restaurants
    restaurant_df = business_df[business_df['categories'].str.contains('Restaurants', na=False)].copy()

    # Process restaurant features
    # Categories
    restaurant_df['categories_list'] = restaurant_df['categories'].str.split(', ').fillna('Unknown')
    mlb = MultiLabelBinarizer()
    categories_encoded = mlb.fit_transform(restaurant_df['categories_list'])
    if categories_encoded.shape[1] > max_categories:
        # Select top categories by frequency
        category_sums = categories_encoded.sum(axis=0)
        top_indices = np.argsort(category_sums)[-max_categories:]
        categories_encoded = categories_encoded[:, top_indices]
        category_names = mlb.classes_[top_indices]
    else:
        category_names = mlb.classes_
    categories_df = pd.DataFrame(categories_encoded, columns=[f'cat_{i}' for i in range(categories_encoded.shape[1])])

    # Parse Business Parking
    def parse_parking(parking_str):
        try:
            parking = eval(parking_str) if isinstance(parking_str, str) else {}
            return [
                parking.get('garage', False),
                parking.get('street', False),
                parking.get('validated', False),
                parking.get('lot', False),
                parking.get('valet', False)
            ]
        except:
            return [0, 0, 0, 0, 0]
    restaurant_df['parking_features'] = restaurant_df['attributes'].apply(
        lambda x: parse_parking(x.get('BusinessParking') if isinstance(x, dict) else None)
    )
    parking_df = pd.DataFrame(restaurant_df['parking_features'].tolist(),
                             columns=['park_garage', 'park_street', 'park_validated', 'park_lot', 'park_valet'])

    # Normalize numerical features
    scaler = StandardScaler()
    restaurant_df[['stars_norm', 'review_count_norm']] = scaler.fit_transform(
        restaurant_df[['stars', 'review_count']]
    )

    # Geospatial features
    restaurant_df['lat_norm'] = scaler.fit_transform(restaurant_df[['latitude']])
    restaurant_df['lon_norm'] = scaler.fit_transform(restaurant_df[['longitude']])

    # Combine restaurant features
    restaurant_features = pd.concat([
        restaurant_df[['business_id', 'stars_norm', 'review_count_norm', 'lat_norm', 'lon_norm']].reset_index(drop=True),
        categories_df.reset_index(drop=True),
        parking_df.reset_index(drop=True)
    ], axis=1)

    # Process user features
    user_df['elite_binary'] = user_df['elite'].apply(lambda x: 1 if x else 0)
    user_df['friends_count'] = user_df['friends'].apply(lambda x: len(x.split(',')) if x != 'None' else 0)
    user_df[['review_count_norm', 'average_stars_norm', 'fans_norm', 'friends_count_norm']] = scaler.fit_transform(
        user_df[['review_count', 'average_stars', 'fans', 'friends_count']]
    )

    # Merge data
    train_data = review_df[['user_id', 'business_id', 'stars']].merge(
        user_df[['user_id', 'review_count_norm', 'average_stars_norm', 'elite_binary', 'fans_norm', 'friends_count_norm']],
        on='user_id'
    ).merge(
        restaurant_features,
        on='business_id'
    )

    return train_data, restaurant_features, user_df, category_names

# --- Two-Tower Model ---
def build_two_tower_model(user_feature_dim, rest_feature_dim, category_dim, embedding_dim):
    # User Tower
    user_inputs = {
        'review_count': tf.keras.Input(shape=(1,), name='review_count', dtype=tf.float32),
        'average_stars': tf.keras.Input(shape=(1,), name='average_stars', dtype=tf.float32),
        'fans': tf.keras.Input(shape=(1,), name='fans', dtype=tf.float32),
        'friends_count': tf.keras.Input(shape=(1,), name='friends_count', dtype=tf.float32),
        'elite': tf.keras.Input(shape=(1,), name='elite', dtype=tf.int32)
    }
    user_dense = layers.Concatenate()([
        user_inputs['review_count'],
        user_inputs['average_stars'],
        user_inputs['fans'],
        user_inputs['friends_count']
    ])
    user_dense = layers.Dense(128, activation='relu')(user_dense)
    user_dense = layers.Dense(64, activation='relu')(user_dense)
    user_elite = layers.Embedding(input_dim=2, output_dim=8)(user_inputs['elite'])
    user_elite = layers.Flatten()(user_elite)
    user_features = layers.Concatenate()([user_dense, user_elite])
    user_embedding = layers.Dense(embedding_dim, activation='relu', name='user_embedding')(user_features)

    # Item Tower
    rest_inputs = {
        'stars': tf.keras.Input(shape=(1,), name='stars', dtype=tf.float32),
        'review_count': tf.keras.Input(shape=(1,), name='review_count_rest', dtype=tf.float32),
        'lat': tf.keras.Input(shape=(1,), name='lat', dtype=tf.float32),
        'lon': tf.keras.Input(shape=(1,), name='lon', dtype=tf.float32),
        'categories': tf.keras.Input(shape=(category_dim,), name='categories', dtype=tf.float32),
        'parking': tf.keras.Input(shape=(5,), name='parking', dtype=tf.float32)
    }
    rest_dense = layers.Concatenate()([
        rest_inputs['stars'],
        rest_inputs['review_count'],
        rest_inputs['lat'],
        rest_inputs['lon'],
        rest_inputs['parking']
    ])
    rest_dense = layers.Dense(128, activation='relu')(rest_dense)
    rest_dense = layers.Dense(64, activation='relu')(rest_dense)
    rest_categories = layers.Dense(32, activation='relu')(rest_inputs['categories'])
    rest_features = layers.Concatenate()([rest_dense, rest_categories])
    rest_embedding = layers.Dense(embedding_dim, activation='relu', name='rest_embedding')(rest_features)

    # Interaction
    score = layers.Dot(axes=1)([user_embedding, rest_embedding])
    output = layers.Dense(1, activation='linear')(score)

    model = Model(inputs=[user_inputs, rest_inputs], outputs=output)
    return model

# --- Baseline: Matrix Factorization ---
def matrix_factorization(train_data, n_factors=50, n_epochs=20):
    from implicit.als import AlternatingLeastSquares
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

# --- Evaluation Metrics ---
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    precision_at_5 = np.mean([1 if pred >= 4 and true >= 4 else 0 for pred, true in zip(y_pred, y_true)])
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Precision@5: {precision_at_5:.4f}")
    return rmse, mae, precision_at_5

# --- Main Workflow ---
def main():
    # Load and preprocess data
    train_data, restaurant_features, user_df, category_names = load_and_preprocess_data(
        'business.json', 'user.json', 'review.json'
    )

    # Split data
    train, test = train_test_split(train_data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)

    # Prepare datasets
    def create_dataset(data):
        user_features = {
            'review_count': data['review_count_norm'].values,
            'average_stars': data['average_stars_norm'].values,
            'fans': data['fans_norm'].values,
            'friends_count': data['friends_count_norm'].values,
            'elite': data['elite_binary'].values
        }
        rest_features = {
            'stars': data['stars_norm'].values,
            'review_count': data['review_count_norm_y'].values,
            'lat': data['lat_norm'].values,
            'lon': data['lon_norm'].values,
            'categories': data[[f'cat_{i}' for i in range(len(category_names))]].values,
            'parking': data[['park_garage', 'park_street', 'park_validated', 'park_lot', 'park_valet']].values
        }
        return user_features, rest_features, data['stars'].values

    train_user, train_rest, train_labels = create_dataset(train)
    val_user, val_rest, val_labels = create_dataset(val)
    test_user, test_rest, test_labels = create_dataset(test)

    # Hyperparameter tuning
    param_grid = {
        'embedding_dim': [32, 64],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64]
    }
    best_rmse = float('inf')
    best_params = {}
    best_model = None

    for params in product(*param_grid.values()):
        emb_dim, lr, batch = params
        print(f"Testing params: emb_dim={emb_dim}, lr={lr}, batch={batch}")
        
        model = build_two_tower_model(
            user_feature_dim=4,  # Numerical features
            rest_feature_dim=4+5,  # Numerical + parking
            category_dim=len(category_names),
            embedding_dim=emb_dim
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                     loss='mse',
                     metrics=['mae'])
        
        history = model.fit(
            [train_user, train_rest], train_labels,
            validation_data=([val_user, val_rest], val_labels),
            epochs=10,
            batch_size=batch,
            verbose=0
        )
        
        val_pred = model.predict([val_user, val_rest], batch_size=batch, verbose=0)
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_pred))
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = {'embedding_dim': emb_dim, 'learning_rate': lr, 'batch_size': batch}
            best_model = model
        
        tf.keras.backend.clear_session()

    print(f"Best params: {best_params}, Validation RMSE: {best_rmse:.4f}")

    # Evaluate best model
    test_pred = best_model.predict([test_user, test_rest], batch_size=best_params['batch_size'], verbose=0)
    evaluate_model(test_labels, test_pred.flatten(), "Two-Tower Model")

    # Baseline: Matrix Factorization
    predict_mf, user_map, item_map = matrix_factorization(train_data)
    mf_pred = [predict_mf(row['user_id'], row['business_id'], user_map, item_map) for _, row in test.iterrows()]
    evaluate_model(test_labels, mf_pred, "Matrix Factorization")

    # Baseline: Item-Based CF
    predict_icf, item_similarities = item_based_cf(train)
    icf_pred = [predict_icf(row['user_id'], row['business_id'], train, item_similarities) for _, row in test.iterrows()]
    evaluate_model(test_labels, icf_pred, "Item-Based CF")

    # Save model
    best_model.save('two_tower_model.h5')

if __name__ == "__main__":
    main()