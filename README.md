# Yelp Restaurant Recommendation System

## Overview

This project, developed by Team Yelpers (Jimmy Chung, Khai Nguyen, Shreyas Kulkarni), aims to simplify restaurant discovery on Yelp by recommending top restaurants in a specific area based on user preferences such as cuisine type (e.g., Korean, Vietnamese), price, menu items, parking availability, and social connections. We leverage the Yelp Open Dataset to build and evaluate multiple recommendation algorithms.

## Dataset

The project uses the Yelp Open Dataset, accessible at Yelp Open Dataset. The dataset includes the following files with relevant details:

| File | Number of Instances | Description | Relevant Columns/Features |
| --- | --- | --- | --- |
| Business.json | 150,346 | Unique businesses | postal_code, latitude, longitude, stars, review_count, Business Parking, categories, hours |
| Checkin.json | 131,930 | Check-ins for businesses | date |
| Review.json | 6,990,280 | Restaurant reviews | user_id, business_id, stars, date, useful, funny, cool, text |
| User.json | 1,987,897 | User information and social connections | review_count, friends, yelping_since, fans, elite, average_stars, compliments (summation) |
| Tip.json | 908,915 | Tips and their counts | date, compliment |

Note: Images from the dataset are not used in this project.

## Methodology

We implement three distinct recommendation algorithms to provide personalized restaurant recommendations:

1. **Social Graph-Based Recommendation (Khai Nguyen)**

   - Utilizes the `user.json` file to leverage the social connections (friends) of users for recommendations.
   - Recommends restaurants based on preferences and activities of a user's friends.

2. **Hybrid Approach (Jimmy Chung)**

   - Combines content-based filtering, collaborative filtering, and/or matrix factorization.
   - Considers restaurant attributes (e.g., cuisine type, price, parking) and user ratings to generate recommendations.

3. **Two-Tower Model (Shreyas Kulkarni)**

   - Implements a two-tower neural network architecture for recommendation.
   - Encodes user and restaurant features separately to predict compatibility.

### Hyperparameter Tuning

Each algorithm undergoes hyperparameter tuning to optimize performance and ensure the best recommendations.

### Evaluation

Models are evaluated using metrics such as **Root Mean Squared Error (RMSE)** to determine the most effective approach.

## Project Goals

- Provide personalized restaurant recommendations based on:
  - Popularity
  - Price
  - Menu items and cuisine type
  - Parking availability
  - Social connections
- Simplify the Yelp experience by reducing the overwhelming number of choices for users.

## Team

- **Jimmy Chung**: Hybrid recommendation system (content-based, collaborative filtering, matrix factorization)
- **Khai Nguyen**: Social graph-based recommendation system
- **Shreyas Kulkarni**: Two-tower model for recommendations

## Setup and Installation

1. **Download the Dataset**:

   - Obtain the Yelp Open Dataset from Yelp Open Dataset.
   - Place the dataset files (`Business.json`, `Checkin.json`, `Review.json`, `User.json`, `Tip.json`) in the project directory.

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   (Ensure you have Python 3.x installed. The `requirements.txt` file will be provided with necessary libraries such as pandas, numpy, scikit-learn, etc.)

## Two-tower model inference

![image](https://github.com/user-attachments/assets/3170d007-af47-4716-95d2-8b66752d70a2)


## Future Work

- Incorporate additional features such as real-time check-in data or user reviews for enhanced recommendations.
- Explore advanced deep learning models for improved accuracy.
- Deploy the recommendation system as a web or mobile application.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
