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

## Notes

1. The data was too large to upload.
   - Training data is around 1GB
   - Testing data is around 340MB
   - Validation data is around 271 MB
2. The data preprocessing will take some time due to large size.
3. I tried working on the Item-based CF as another baseline but I ran out of memory as it was not so computationlly efficient for me.
4. The current project uses Two Tower model with rmse around 1.13 and baseline Matrix Factoriation with rmse around 3.93.

![image](https://github.com/user-attachments/assets/f64761e5-3182-4c74-b985-028c34b6f90e)

## License

This project is licensed under the MIT License.
