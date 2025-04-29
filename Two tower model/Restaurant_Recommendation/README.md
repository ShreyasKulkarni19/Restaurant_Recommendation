# Restaurant Recommendation System

This project implements a restaurant recommendation system using various machine learning models. The system utilizes user and business data to provide personalized restaurant recommendations.

## Project Structure

```
Restaurant_Recommendation
├── data
│   ├── business.json       # Business data for restaurants
│   ├── user.json           # User data relevant to recommendations
│   └── review.json         # Review data for restaurants
├── src
│   ├── __init__.py         # Marks the src directory as a Python package
│   ├── data_preprocessing.py # Functions for loading and preprocessing data
│   ├── models
│   │   ├── __init__.py     # Marks the models directory as a Python package
│   │   ├── two_tower_model.py # Implementation of the two-tower recommendation model
│   │   └── baselines.py     # Implementations of baseline recommendation algorithms
│   ├── evaluation.py        # Functions for evaluating model performance
│   └── main.py             # Entry point of the application
├── requirements.txt         # Lists project dependencies
└── README.md                # Documentation for the project
```

## Setup Instructions

1. Clone the repository:

   ```
   git clone <repository-url>
   cd Restaurant_Recommendation
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Prepare the data files (`business.json`, `user.json`, `review.json`) in the `data` directory.

## Usage

To run the recommendation system, execute the following command:

```
python src/main.py
```

This will load the data, train the models, and evaluate their performance.

## Data Preprocessing

The data preprocessing step is crucial for preparing the raw data into a format suitable for training machine learning models. The following steps are performed during data preprocessing:

1. **Loading Data**:

   - The raw data files (`business.json`, `user.json`, `review.json`, etc.) are loaded using the `load_large_json` function, which handles large JSON files in chunks to avoid memory issues.

2. **Filtering Restaurants**:

   - The `business.json` file is filtered to include only businesses categorized as restaurants.

3. **Processing Categories**:

   - The `categories` field in the restaurant data is split into individual categories.
   - A `MultiLabelBinarizer` is used to encode the categories into a one-hot encoded format.
   - Only the top `max_categories` categories are retained based on their frequency.

4. **Processing Parking Features**:

   - The `BusinessParking` attribute is parsed to extract parking-related features such as garage, street, validated, lot, and valet parking availability.

5. **Normalizing Features**:

   - Numerical features such as `stars`, `review_count`, `latitude`, and `longitude` are normalized using `StandardScaler` to ensure they are on a similar scale.

6. **Processing User Features**:

   - User-related features such as `review_count`, `average_stars`, `fans`, and `friends_count` are normalized.
   - The `elite` field is converted into a binary feature indicating whether a user is elite or not.

7. **Merging Data**:

   - The processed user, restaurant, and review data are merged to create a single dataset for training.

8. **Saving Preprocessed Data**:

   - The preprocessed data is saved to disk using the `save_preprocessed_data` function for reuse in subsequent steps.

9. **Loading Preprocessed Data**:
   - If preprocessed data already exists, it can be loaded using the `load_preprocessed_data` function to save time.

These steps ensure that the data is clean, consistent, and ready for use in the recommendation system.

## Experiments

The experiments conducted in this project aimed to evaluate and improve the performance of the restaurant recommendation system. Below is an overview of the experiments:

### 1. Baseline Model Setup

- We started by implementing baseline models to establish a reference point for performance evaluation.
- The baseline models included:
  - **Matrix Factorization**: A collaborative filtering approach that factorizes the user-item interaction matrix into latent factors.
  - **Item-Based Collaborative Filtering**: A similarity-based approach that recommends items similar to those the user has interacted with.
- These models were evaluated using metrics such as RMSE and MAE to understand their effectiveness.

### 2. Two-Tower Model

- After establishing the baseline, we implemented the Two-Tower Model, a deep learning-based approach designed to capture user and item features more effectively.
- The model consists of two separate towers:
  - **User Tower**: Processes user-related features such as review count, average stars, and elite status.
  - **Item Tower**: Processes item-related features such as restaurant categories, parking availability, and geographical location.
- The outputs of the two towers are combined to predict user-item interactions.

### 3. Hyperparameter Tuning

- To optimize the performance of the Two-Tower Model, we conducted a grid search over the following hyperparameters:
  - **Embedding Dimension**: Tested values such as 32, 64, and 128.
  - **Learning Rate**: Tested values such as 0.001 and 0.0001.
  - **Batch Size**: Tested values such as 32 and 64.
- The grid search involved training the model with different combinations of these hyperparameters and evaluating their performance on the validation set.
- The best combination of hyperparameters was selected based on the lowest validation RMSE.

### 4. Final Model Evaluation

- The final Two-Tower Model, trained with the best hyperparameters, was evaluated on the test set.
- The performance of the final model was compared with the baseline models to demonstrate its effectiveness.

These experiments highlight the iterative process of model development and optimization, leading to a robust recommendation system.

## License

This project is licensed under the MIT License.
