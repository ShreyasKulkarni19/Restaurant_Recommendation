# filepath: Restaurant_Recommendation/src/main.py
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from models.two_tower_model import build_two_tower_model
from models.baselines import matrix_factorization
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from itertools import product
import numpy as np

def load_large_json(file_path, chunksize=10000):
    """Load large JSON files in chunks to avoid memory issues."""
    print(f"Loading {file_path} in chunks...")
    chunks = []
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def main():
    print("Loading data files...")
    try:
        business_data = load_large_json('../data/business.json')
        user_data = load_large_json('../data/user.json')
        tip_data = load_large_json('../data/tip.json')
        checkin_data = load_large_json('../data/checkin.json')
        review_data = load_large_json('../data/review.json')
        print("Data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except MemoryError as e:
        print(f"MemoryError: {e}")
        return

    print("Preprocessing data...")
    train_data, restaurant_features, user_df, category_names = load_and_preprocess_data(
        business_data, user_data, review_data, tip_data, checkin_data, max_categories=50
    )
    print("Data loaded and preprocessed.")
    print("-------------------------------------------------------------------")

    train, test = train_test_split(train_data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    print("Data split into train, validation, and test sets.")
    print(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")
    print("-------------------------------------------------------------------")

    def create_dataset(data):
        user_features = {
            'review_count': data['review_count_norm_x'].values.astype(np.float32),
            'average_stars': data['average_stars_norm'].values.astype(np.float32),
            'fans': data['fans_norm'].values.astype(np.float32),
            'friends_count': data['friends_count_norm'].values.astype(np.float32),
            'elite': data['elite_binary'].values.astype(np.float32)
        }

        rest_features = {
            # Removed 'stars' from here since it's what we're trying to predict
            'review_count': data['review_count_norm_y'].values.astype(np.float32),
            'lat': data['lat_norm'].values.astype(np.float32),
            'lon': data['lon_norm'].values.astype(np.float32),
            'categories': data[[f'cat_{i}' for i in range(50)]].values.astype(np.float32),
            'parking': data[['park_garage', 'park_street', 'park_validated', 'park_lot', 'park_valet']].values.astype(np.float32)
        }

        labels = data['stars'].values.astype(np.float32)
        return user_features, rest_features, labels

    train_user, train_rest, train_labels = create_dataset(train)
    val_user, val_rest, val_labels = create_dataset(val)
    test_user, test_rest, test_labels = create_dataset(test)

    print("Hyperparameter tuning for Two-Tower Model...")
    param_grid = {
        'embedding_dim': [64],
        'learning_rate': [0.001],
        'batch_size': [64]
    }
    best_rmse = float('inf')
    best_params = {}
    best_model = None

    for params in product(*param_grid.values()):
        emb_dim, lr, batch = params
        print(f"Testing params: emb_dim={emb_dim}, lr={lr}, batch={batch}")

        model = build_two_tower_model(
            user_feature_dim=4,
            rest_feature_dim=4+5,
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

    print("Evaluating best model on test set...")
    test_pred = best_model.predict([test_user, test_rest], batch_size=best_params['batch_size'], verbose=0)
    evaluate_model(test_labels, test_pred.flatten(), "Two-Tower Model")
    print("-------------------------------------------------------------------")
    print("Evaluating baseline models...")
    print("Evaluating Matrix Factorization...")

    predict_mf, user_map, item_map = matrix_factorization(train_data)
    mf_pred = [predict_mf(row['user_id'], row['business_id'], user_map, item_map) for _, row in test.iterrows()]
    evaluate_model(test_labels, mf_pred, "Matrix Factorization")

    print("-------------------------------------------------------------------")
    print("Training completed for all models.")
    print("Model saved as 'two_tower_model.h5'")
    best_model.save('two_tower_model.h5')

if __name__ == "__main__":
    main()