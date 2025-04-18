from tensorflow.keras import layers, Model
import tensorflow as tf

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