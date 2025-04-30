import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import os
import pickle

def save_preprocessed_data(data, file_path):
    """
    Save preprocessed data to disk using pickle.

    Args:
        data: The preprocessed data to save.
        file_path: The file path where the data should be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {file_path}")

def load_preprocessed_data(file_path):
    """
    Load preprocessed data from disk if it exists.

    Args:
        file_path: The file path from where the data should be loaded.

    Returns:
        The loaded data if the file exists, otherwise None.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Data loaded from {file_path}")
        return data
    else:
        print(f"No preprocessed data found at {file_path}")
        return None

def load_and_preprocess_data(business_file, user_file, review_file, tip_file=None, checkin_file=None, max_categories=50):
    business_df = business_file
    user_df = user_file
    review_df = review_file

    # Process tip_file if provided
    if tip_file is not None:
        tip_df = tip_file
        tip_features = tip_df.groupby('business_id')['compliment_count'].sum().reset_index()
        business_df = business_df.merge(tip_features, on='business_id', how='left')
        business_df['compliment_count'] = business_df['compliment_count'].fillna(0)
    else:
        business_df['compliment_count'] = 0

    # Process checkin_file if provided
    if checkin_file is not None:
        checkin_df = checkin_file
        checkin_df['checkin_count'] = checkin_df['date'].str.split(',').apply(len)
        checkin_features = checkin_df[['business_id', 'checkin_count']]
        business_df = business_df.merge(checkin_features, on='business_id', how='left')
        business_df['checkin_count'] = business_df['checkin_count'].fillna(0)
    else:
        business_df['checkin_count'] = 0

    # Filter restaurants
    restaurant_df = business_df[business_df['categories'].str.contains('Restaurants', na=False)].copy()

    # Process categories
    restaurant_df['categories_list'] = restaurant_df['categories'].str.split(', ').fillna('Unknown')
    mlb = MultiLabelBinarizer()
    categories_encoded = mlb.fit_transform(restaurant_df['categories_list'])
    if categories_encoded.shape[1] > max_categories:
        category_sums = categories_encoded.sum(axis=0)
        top_indices = np.argsort(category_sums)[-max_categories:]
        categories_encoded = categories_encoded[:, top_indices]
        category_names = mlb.classes_[top_indices]
    else:
        category_names = mlb.classes_
    categories_df = pd.DataFrame(categories_encoded, columns=[f'cat_{i}' for i in range(categories_encoded.shape[1])])

    # Process parking features
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

    # Normalize features
    scaler = StandardScaler()
    restaurant_df[['stars_norm', 'review_count_norm', 'compliment_count_norm', 'checkin_count_norm']] = scaler.fit_transform(
        restaurant_df[['stars', 'review_count', 'compliment_count', 'checkin_count']]
    )
    restaurant_df['lat_norm'] = scaler.fit_transform(restaurant_df[['latitude']])
    restaurant_df['lon_norm'] = scaler.fit_transform(restaurant_df[['longitude']])

    # Combine restaurant features - REMOVE stars_norm from features as it's the target
    restaurant_features = pd.concat([ 
        restaurant_df[['business_id', 'review_count_norm', 'compliment_count_norm', 'checkin_count_norm', 'lat_norm', 'lon_norm']].reset_index(drop=True),
        categories_df.reset_index(drop=True),
        parking_df.reset_index(drop=True)
    ], axis=1)

    # Process user features
    user_df['elite_binary'] = user_df['elite'].apply(lambda x: 1 if x else 0)
    user_df['friends_count'] = user_df['friends'].apply(lambda x: len(x.split(',')) if x != 'None' else 0)
    user_df[['review_count_norm', 'average_stars_norm', 'fans_norm', 'friends_count_norm']] = scaler.fit_transform(
        user_df[['review_count', 'average_stars', 'fans', 'friends_count']]
    )

    # Merge data for training
    train_data = review_df[['user_id', 'business_id', 'stars']].merge(
        user_df[['user_id', 'review_count_norm', 'average_stars_norm', 'elite_binary', 'fans_norm', 'friends_count_norm']],
        on='user_id'
    ).merge(
        restaurant_features,
        on='business_id'
    )

    return train_data, restaurant_features, user_df, category_names