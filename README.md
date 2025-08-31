# Personality-Prediction-MBTi
MBTI Personality Prediction Using Machine Learning
## Project Overview
This project aims to predict Myers-Briggs Type Indicator (MBTI) personality types based on user posts from social media or similar sources. The analysis leverages Natural Language Processing (NLP) techniques for text preprocessing and multiple machine learning algorithms for classification of personality traits.

## Dataset
Dataset contains 8,675 users with posts and corresponding MBTI personality types.

Each user has an associated MBTI type (16 possible) and a collection of posts.

The dataset is explored for missing values (none found) and data distribution across personality types.

## Data Preprocessing
Text cleaning includes removal of URLs, punctuation, stopwords, and special MBTI personality keywords.

Tokenization, lemmatization, and case normalization applied for preparing text data.

Posts are split into individual texts and aggregated features include:

Number of words per post

Variance of word counts across posts

MBTI personality types are converted into binary vectors representing four dichotomies: I/E, N/S, T/F, J/P.

## Feature Engineering
Text is vectorized using:

Count Vectorizer (limited to top 1000 features with specified document frequency thresholds)

TF-IDF transformation applied on count vectors for weighted text representation.

Reverse dictionary mapping used to identify most common features.

## Model Training and Evaluation
The data is split into training and test sets for each MBTI trait.

Models used include:

Random Forest Classifier

Gradient Descent (SGDClassifier)

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

XGBoost Classifier

Accuracy scores are computed for each personality dichotomy (I/E, N/S, T/F, J/P).

Logistic Regression showed highest accuracy in the initial evaluation (~58% overall).

## Visualizations
Distribution of posts across MBTI personality types using bar plots.

Word cloud visualizations representing common words across posts by personality type.

Correlation heatmap between MBTI trait features.

## Usage
Clone the repository and place the dataset (mbti_1.csv) in the project folder.

Install required libraries:

text
pip install numpy pandas matplotlib seaborn scikit-learn nltk xgboost wordcloud
Run preprocessing to clean and transform text data.

Use vectorizers to convert text into numerical features.

Train and evaluate models on the dataset.

Visualize the results and interpret model performances.

## Dependencies
Python 3.x

Pandas, Numpy

Matplotlib, Seaborn

Scikit-learn

NLTK (for text processing)

XGBoost (for gradient boosting models)

WordCloud (for visualization)

## Future Work
Experiment with deep learning models for text (e.g., LSTM, transformers).

Use cross-validation and hyperparameter tuning for improved model accuracy.

Explore other NLP features like sentiment, POS tagging, or embeddings.
