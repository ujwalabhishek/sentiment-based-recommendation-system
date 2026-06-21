# Sentiment-Based Product Recommendation System

This project builds an end-to-end sentiment-based product recommendation system for an e-commerce company named **Ebuss**. The goal is to improve product recommendations by combining collaborative filtering with sentiment analysis on historical user reviews and ratings.

## Business Context

Ebuss sells products across categories such as household essentials, books, personal care, medicines, cosmetics, beauty products, electrical appliances, kitchen and dining, and health care. To compete with major e-commerce platforms, Ebuss wants better personalized product recommendations for users based on their past reviews and ratings.

As a Machine Learning Engineer, the task is to build and deploy a recommendation system that:

1. Analyzes customer product reviews and ratings.
2. Builds a sentiment classification model.
3. Builds and evaluates recommendation systems.
4. Improves recommendations using the sentiment model.
5. Deploys the complete system with a user interface.

## Dataset

The project uses a subset of a product reviews dataset inspired by the Kaggle Grammar and Online Product Reviews dataset.

- Product reviews dataset: https://cdn.upgrad.com/uploads/production/c2504c0d-6080-4e1e-8d4c-852b3e68a0ed/sample30.csv
- Attribute description: https://cdn.upgrad.com/uploads/production/a2446a81-154b-49cf-8fb2-2e87614496e6/Data+Attribute+Description.csv

Dataset summary:

- 30,000 product reviews
- More than 200 products
- More than 20,000 users
- `reviews_username` identifies users for recommendation

## Project Requirements

### 1. Data Sourcing and Sentiment Analysis

Perform the following steps on the product review data:

1. Exploratory data analysis
2. Data cleaning
3. Text preprocessing
4. Feature extraction using one or more methods such as:
   - Bag of words
   - TF-IDF vectorization
   - Word embeddings
5. Train at least three sentiment classification models from the following options:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Naive Bayes

The final sentiment model should be selected based on model performance. If required, handle class imbalance and perform hyperparameter tuning.

### 2. Recommendation System

Build and evaluate both recommendation approaches:

1. User-based recommendation system
2. Item-based recommendation system

Select the recommendation system best suited for this case study based on evaluation results. The selected system should recommend the top 20 products a user is most likely to purchase based on historical ratings.

### 3. Sentiment-Based Recommendation Reranking

After generating 20 candidate product recommendations for a user, use the selected sentiment analysis model to filter and rank those products.

Final output:

- Recommend the best 5 products from the 20 candidates.
- The final 5 should be selected using sentiment scores from reviews of the recommended products.

### 4. Flask Deployment

Deploy the end-to-end project using Flask.

The user interface must:

1. Accept an existing username as input.
2. Provide a submit button.
3. Display 5 recommended products for the entered username.

Important assumption:

- No new users or products will be introduced.
- Sentiment analysis and recommendation predictions are only required for users and products already present in the dataset.

## Expected Deliverables

Submit the following for evaluation:

1. An end-to-end Jupyter Notebook containing:
   - Data cleaning
   - Text preprocessing
   - Feature extraction
   - Sentiment analysis model training and evaluation
   - User-based recommendation system
   - Item-based recommendation system
   - Recommendation system evaluation
   - Final sentiment-based recommendation logic
2. Deployment files:
   - `model.py`: contains only the selected ML model, selected recommendation system, and the logic required to deploy the project.
   - `app.py`: Flask application connecting the backend model logic with the frontend.
   - HTML UI code, either as a template file or embedded through Flask rendering logic.

The final ML model and recommendation system should be defined and initialized directly inside `model.py`.

## Reference Notebook Guidance

The provided recommendation notebook covers:

- Train/test split for recommendation evaluation
- Dummy train and dummy test matrices
- User similarity matrix
- Cosine similarity
- Adjusted cosine similarity
- User-user prediction and evaluation
- Item-item prediction and evaluation
- RMSE-based evaluation for rated products
- Top recommendation generation for a user