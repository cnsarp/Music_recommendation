import streamlit as st
from PIL import Image
from io import BytesIO, StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import os

data_dir = os.path.join(os.path.dirname(__file__), 'data')

#DATASETS
SPOTIFY_DATASET_PATH = "src/streamlit/data/spotify_dataset.csv"
Nowplaying_DATASET_PATH = "src/streamlit/data/module4_cleaned.csv"
TRACKS_PATH ="src/streamlit/data/tracks.csv" 
IMAGE_PATH = "src/streamlit/data/music_5.jpg"
GENRES_PATH = "src/streamlit/data/df_genres_5000.csv"
SCHEMA_PATH = "src/streamlit/data/schemarecommendation.png"
PCA_PATH = "src/streamlit/data/PCA.png"
Sentiment_dis= "src/streamlit/data/Sentiment_count.png"
np_features_dis="src/streamlit/data/continuous features.png"
np_corr= "src/streamlit/data/np_correlation_matrix.png"
Feature_importance= "src/streamlit/data/Feature importance_main.png"
np_deep= "src/streamlit/data/deep_learning_model loss_model acurracy.png"
Track_per_genre ="src/streamlit/data/track_genre_pop.png" 
Spotify_dis = "src/streamlit/data/Spotify_dis.png"
Spotify_cor = "src/streamlit/data/spotify_correlation.png"
Genre_Example = "src/streamlit/data/genre_example.png"


#DATASETS:

spotify = pd.read_csv("src/streamlit/data/spotify_dataset.csv", index_col=0)

# Set page configuration
st.set_page_config(page_title="Music Recommendations")

@st.cache_data
def resize_image(image_path, width):
    img = Image.open(image_path)
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio)
    resized_img = img.resize((width, height))
    return resized_img


resized_image = resize_image(IMAGE_PATH, 900)  # Adjust width here

#This is for styling the sidebar
st.markdown(
    """
    <style>
    /* Sidebar text size */
    [data-testid="stMarkdownContainer"]p{
        font-size: 30px;
    }
    /* Sidebar header padding */
    [data-testid="stSidebarHeader"] {
        padding-top: 150px;
    }
    [data-testid="stSidebar"]{
        background-color: #87A96B
    }
    html {
        font-size: 20px;
        font-weight: bold;
        }
            
    [data-testid="stRadio"]{
       font-size: inherit;
       font-weight: inherit;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)



# Sidebar for navigation
st.sidebar.title("Table of Contents")
section = st.sidebar.radio("Go to", [
    "Project Overview",
    "Datasets",
    "Data Preprocessing",
    "Exploratory Data Analysis (EDA)",
    "Modeling",
    "Analysis of the Best Model",
    "Deep Learning",
    "Genre Detection",
    "Recommendation",
    "Conclusion and Critism"
])

# Dummy data and metrics for display purposes
base_logistic_regression_report = """
Base Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      1451
           1       1.00      1.00      1.00    452048

    accuracy                           1.00    453499
   macro avg       0.50      0.50      0.50    453499
weighted avg       0.99      1.00      1.00    453499
"""


base_logistic_regression_confusion_matrix = np.array([[0, 1451], [0, 452048]])

logistic_regression_report = """
Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       0.01      0.77      0.02      1451
           1       1.00      0.77      0.87    452048

    accuracy                           0.77    453499
   macro avg       0.50      0.77      0.45    453499
weighted avg       1.00      0.77      0.87    453499
"""

logistic_regression_confusion_matrix = np.array([[1119, 332], [102593, 349455]])

decision_tree_report = """
Decision Tree Classification Report:
              precision    recall  f1-score   support

           0       0.28      0.94      0.43      1451
           1       1.00      0.99      1.00    452048

    accuracy                           0.99    453499
   macro avg       0.64      0.97      0.72    453499
weighted avg       1.00      0.99      0.99    453499
"""

decision_tree_confusion_matrix = np.array([[1368, 83], [3472, 448576]])

random_forest_report = """
Random Forest Classification Report:
              precision    recall  f1-score   support

           0       0.29      0.94      0.44      1451
           1       1.00      0.99      1.00    452048

    accuracy                           0.99    453499
   macro avg       0.64      0.97      0.72    453499
weighted avg       1.00      0.99      0.99    453499
"""

random_forest_confusion_matrix = np.array([[1366, 85], [3348, 448700]])

xgboost_report = """
XGBoost Classification Report:
              precision    recall  f1-score   support

           0       0.23      0.95      0.38      1451
           1       1.00      0.99      0.99    452048

    accuracy                           0.99    453499
   macro avg       0.62      0.97      0.69    453499
weighted avg       1.00      0.99      0.99    453499
"""

xgboost_confusion_matrix = np.array([[1372, 79], [4473, 447575]])

best_model_report = """
Classification Report:
               precision    recall  f1-score   support

           0       0.24      0.32      0.27      1451
           1       1.00      1.00      1.00    452048

    accuracy                           0.99    453499
   macro avg       0.62      0.66      0.64    453499
weighted avg       1.00      0.99      0.99    453499
"""

best_model_confusion_matrix = np.array([[470, 981], [1514, 450534]])

# Optimized Random Forest with Class Weights data
class_weights_report = """
Classification Report:
               precision    recall  f1-score   support

           0       0.35      0.92      0.50      1451
           1       1.00      0.99      1.00    452048

    accuracy                           0.99    453499
   macro avg       0.67      0.96      0.75    453499
weighted avg       1.00      0.99      1.00    453499
"""

class_weights_confusion_matrix = np.array([[1330, 121], [2510, 449538]])

# Deep Learning model results data
deep_learning_report = """
Classification Report:
               precision    recall  f1-score   support

           0       0.13      0.92      0.23      1451
           1       1.00      0.98      0.99    452048

    accuracy                           0.98    453499
   macro avg       0.57      0.95      0.61    453499
weighted avg       1.00      0.98      0.99    453499
"""

best_model_content_features = """
Classification Report:
               precision    recall  f1-score   support

           0       0.21      0.95      0.35     1451
           1       1.00      0.99      0.99    452048

    accuracy                           0.99    453499
   macro avg       0.61      0.97      0.67    453499
weighted avg       1.00      0.99      0.99    453499
"""



deep_learning_confusion_matrix = np.array([[1333, 118], [8733, 443315]])

# Function to display confusion matrix with proper labels
def display_confusion_matrix(matrix):
    df_cm = pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    st.dataframe(df_cm)
    

# Project Overview
if section == "Project Overview":
    st.image(resized_image)
    st.title("Music Recommendation")
    st.markdown("""
    ## Project Objective
    The objective of this project is to develop a music recommendation system that predicts the perceived appreciation of music based on Twitter analyses and user feedback.
    We utilize contextual data from the nowplaying-RS dataset and the Spotify Tracks dataset.
    ## Data Sources
    1. **nowplaying-RS Dataset**: Contains 11.6 million music listening events with rich contextual and content features.
    2. **Spotify Tracks Dataset**: Includes audio features and genre information for Spotify tracks.
    """)
    

# Dataset
elif section == "Datasets":
    st.header("Datasets")

    st.markdown("""
    ## nowplaying-RS Dataset
    The nowplaying-RS dataset consists of three primary datasets that we will examine to describe their structures:
    """)

    st.subheader("1. sentiment_values.csv")
    st.markdown("""
    The “sentiment_values.csv” dataset provides sentiment analysis results for hashtags encountered in the listening events. It includes sentiment scores from various lexicons (Vader, AFINN, Opinion Lexicon, SentiStrength) such as minimum, maximum, sum, and average scores. The hashtag column links sentiment scores to specific hashtags associated with the listening events:
    """)
    sentiment_values_columns = {
        "hashtag": "Links sentiment scores to specific hashtags",
        "vader_min": "Minimum Vader sentiment score",
        "vader_max": "Maximum Vader sentiment score",
        "vader_sum": "Sum of Vader sentiment scores",
        "vader_avg": "Average Vader sentiment score",
        "afinn_min": "Minimum AFINN sentiment score",
        "afinn_max": "Maximum AFINN sentiment score",
        "afinn_sum": "Sum of AFINN sentiment scores",
        "afinn_avg": "Average AFINN sentiment score",
        "ol_min": "Minimum Opinion Lexicon sentiment score",
        "ol_max": "Maximum Opinion Lexicon sentiment score",
        "ol_sum": "Sum of Opinion Lexicon sentiment scores",
        "ol_avg": "Average Opinion Lexicon sentiment score",
        "ss_min": "Minimum SentiStrength sentiment score",
        "ss_max": "Maximum SentiStrength sentiment score",
        "ss_sum": "Sum of SentiStrength sentiment scores",
        "ss_avg": "Average SentiStrength sentiment score"
    }
    sentiment_values_df = pd.DataFrame(list(sentiment_values_columns.items()), columns=["Column", "Description"])
    st.table(sentiment_values_df)

    st.subheader("2. user_track_hashtag_timestamp.csv")
    st.markdown("""
    The “user_track_hashtag_timestamp.csv” dataset captures associations between users, tracks, hashtags, and timestamps of listening events. Key columns include user_id, track_id, hashtag, and created_at. This dataset enables us to explore user engagement and interactions with music content through hashtags on social media:
    """)
    user_track_columns = {
        "user_id": "Unique identifier for users",
        "track_id": "Unique identifier for tracks",
        "hashtag": "Hashtags associated with listening events",
        "created_at": "Timestamp of listening events"
    }
    user_track_df = pd.DataFrame(list(user_track_columns.items()), columns=["Column", "Description"])
    st.table(user_track_df)

    st.subheader("3. context_content_features.csv")
    st.markdown("""
    The "context_content_features.csv" dataset contains contextual and content features related to music listening events collected from Twitter. These features include:
    """)
    context_content_columns = {
        "coordinates": "Geographical coordinates of the listening event",
        "instrumentalness": "Likelihood the track is instrumental",
        "liveness": "Likelihood the track is live",
        "speechiness": "Presence of spoken words in the track",
        "danceability": "Suitability of the track for dancing",
        "valence": "Musical positiveness conveyed by the track",
        "loudness": "Overall loudness of the track in decibels",
        "tempo": "Tempo of the track in beats per minute",
        "acousticness": "Confidence measure of whether the track is acoustic",
        "energy": "Measure of intensity and activity of the track",
        "mode": "Modality of the track (major=1, minor=0)",
        "key": "Key of the track",
        "artist_id": "Unique identifier for artists",
        "place": "Place associated with the tweet",
        "geo": "Geolocation data",
        "tweet_lang": "Language of the tweet",
        "track_id": "Unique identifier for tracks",
        "created_at": "Timestamp of the tweet",
        "lang": "Language of the tweet",
        "time_zone": "Time zone of the user"
    }
    context_content_df = pd.DataFrame(list(context_content_columns.items()), columns=["Column", "Description"])
    st.table(context_content_df)
    
    st.markdown("""
    ## Spotify Tracks Dataset
    The "Spotify Tracks Dataset" provides detailed audio features for tracks across various genres. It can be used for building recommendation systems, classification tasks, and more. Key columns include:
    """)
    spotify_tracks_columns = {
        "track_id": "The Spotify ID for the track",
        "artists": "Names of artists who performed the track",
        "album_name": "Album name in which the track appears",
        "track_name": "Name of the track",
        "popularity": "Popularity of the track (0-100)",
        "duration_ms": "Track length in milliseconds",
        "explicit": "Whether the track has explicit lyrics",
        "danceability": "Suitability of the track for dancing",
        "energy": "Measure of intensity and activity",
        "key": "Key of the track",
        "loudness": "Overall loudness of the track in decibels",
        "mode": "Modality of the track (major=1, minor=0)",
        "speechiness": "Presence of spoken words in the track",
        "acousticness": "Confidence measure of whether the track is acoustic",
        "instrumentalness": "Likelihood the track is instrumental",
        "liveness": "Likelihood the track is live",
        "valence": "Musical positiveness conveyed by the track",
        "tempo": "Tempo of the track in beats per minute",
        "time_signature": "Estimated time signature",
        "track_genre": "Genre of the track"
    }
    spotify_tracks_df = pd.DataFrame(list(spotify_tracks_columns.items()), columns=["Column", "Description"])
    st.table(spotify_tracks_df)
  

# Data Preprocessing
elif section == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    st.subheader("sentiment_values.csv")
    st.markdown("""
    The “sentiment_values.csv” dataset was cleaned using the following steps:
    - **Renaming misplaced columns:** Enhanced interpretability by renaming columns to align with sentiment analysis metrics.
    - **Dropping unnecessary columns:** Removed columns that did not provide valuable information for analysis.
    - **Addressing missing values:** Imputed missing sentiment scores with average values and dropped rows with missing values.
    - **Correlation matrix:** Dropped highly correlated sentiment scores ('vader_score', 'afinn_score', 'ss_score') and retained 'ol_score' as 'sentiment_score'.
    """)

#    st.image("path/to/figure2.png", caption="Figure 2: Correlation Matrix")

    st.subheader("user_track_hashtag_timestamp.csv")
    st.markdown("""
    The “user_track_hashtag_timestamp.csv” dataset was cleaned using the following steps:
    - **Handling Null Values:** Removed null values in the 'hashtag' column.
    - **Filtering Tracks by Usage Count:** Focused on tracks played more than 50 times.
    - **Merging with Cleaned Sentiment Dataset:** Combined with sentiment scores based on the 'hashtag' column.
    - **Confirming Dataset Integrity:** Performed checks to validate the merged dataset.
    """)

    st.subheader("context_content_features.csv")
    st.markdown("""
    The "context_content_features.csv" dataset was cleaned using the following steps:
    - **Loading Necessary Columns:** Loaded only relevant columns to reduce memory usage.
    - **Removing Tracks with Fewer Plays:** Removed tracks played fewer than 50 times.
    - **Dropping Unnecessary Columns & Removing Null Values:** Dropped irrelevant columns and null values.
    - **Filtering English Language Entries:** Included only English language entries.
    - **Merging with Sentiment Data:** Combined with sentiment data for enriched insights.
    - **Converting and Dropping Columns:** Converted certain columns to string type and dropped unnecessary ones.
    - **Filtering USA Time Zones:** Included only USA time zones and simplified their names.
    - **Creating Binary Sentiment Column:** Categorized sentiment as positive or negative based on the sentiment_score.
    - **Reordering Columns:** Arranged columns in a logical sequence for better readability.
    - **One-Hot Encoding Time Zone:** Converted categorical data into numerical format.
    """)
    
    # Load and display preprocessed data
    st.subheader("Preprocessed Data Sample")
    df_1 = pd.read_csv(Nowplaying_DATASET_PATH)  # Replace with the path to your preprocessed data
    st.dataframe(df_1.head())

    st.subheader("Spotify Tracks Dataset")
    st.markdown("""
    The "Spotify Tracks Dataset" was cleaned using the following steps:
    - **Loading the Dataset:** Loaded the dataset with 114000 rows and 20 columns.
    - **Handling Duplicates and Null Values:** Dropped 450 duplicates and removed rows with missing values, resulting in 113549 rows.
    """)

    # Load and display preprocessed data
    st.subheader("Preprocessed Data Sample")
    st.dataframe(spotify.head())

    st.markdown("""
    These preprocessing steps collectively prepare the datasets for analysis or modeling tasks, ensuring data quality, consistency, and relevance for deriving meaningful insights or building predictive models.
    """)

# Exploratory Data Analysis (EDA)
elif section == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Nowplaying-RS Dataset")
    st.markdown("""
    The purpose of this Exploratory Data Analysis (EDA) is to understand the underlying patterns and relationships within the nowplaying-RS dataset. We aim to explore the data, identify and address issues such as multicollinearity and data imbalance, and prepare the dataset for further predictive modeling.
    - **Distribution of Sentiment:** The data is highly imbalanced, with a significant majority of positive sentiments (1). This imbalance can affect model training, leading to biased predictions (Fig. 3).
    - **Feature Distributions:** Visualizing continuous variables shows different distribution patterns, necessitating normalization or scaling before modeling (Fig. 4).
    - **Correlation Analysis:** Identifying potential multicollinearity is crucial. Features like loudness and energy exhibit high correlation, suggesting multicollinearity that could affect model performance (Fig. 5).
    - **Other Notable Correlations:** There is a strong negative correlation between acousticness and energy, and moderate correlations between danceability and valence, as well as acousticness and loudness.
    """)

    st.image(Sentiment_dis, caption="Figure 3: Sentiment Distribution in Nowplaying-RS Dataset")
    st.image(np_features_dis, caption="Figure 4: Distribution of Continuous Variables in Nowplaying-RS Dataset")
    st.image(np_corr, caption="Figure 5: Correlation Matrix for Nowplaying-RS Dataset")

    st.subheader("Spotify Tracks Dataset")
    st.markdown("""
    Similar to the nowplaying-RS dataset, we analyze the Spotify Tracks Dataset to uncover underlying patterns and distributions.
    - **Feature Distributions:** Variables like danceability, valence, and tempo exhibit quasi-normal distribution. However, features such as duration_ms, instrumentalness, and speechiness show unequal distributions with extreme values (Fig. 6).
    - **Sampling Bias:** The dataset contains exactly 1000 entries for each of the 114 genres, which may affect the distribution of some variables, notably energy.
    - **Correlation Analysis:** Similar patterns are observed as in the nowplaying-RS dataset, with strong correlations between energy and loudness, and a negative correlation between acousticness and energy (Fig. 7). Additionally, explicit tracks tend to have higher speechiness.
    - **Genre Popularity:** Visualization of genre popularity reveals how popularity is distributed across different genres, providing insights into genre-specific trends (Fig. 8).
    """)
    
    st.image(Spotify_dis, caption="Figure 6: Feature Distributions in Spotify Tracks Dataset")
    st.image(Spotify_cor, caption="Figure 7: Correlation Matrix for Spotify Tracks Dataset")
    st.image(Track_per_genre, caption="Figure 8: Genre Popularity in Spotify Tracks Dataset")


# Modeling

# Modeling section
elif section == "Modeling":
    st.header("Modeling with Context and Content Features")
    st.subheader("Nowplaying-RS Dataset")
    
    st.markdown("""
    For the first part of this project, we aimed to develop machine learning models to predict music appreciation based on user context and content features.
    
    **Preprocessing and Initial Modeling Efforts:**
    - Data Splitting: The dataset was split into training and testing sets using a standard 80/20 split.
    - Feature Scaling: StandardScaler was applied to normalize feature values.

    **Logistic Regression Baseline Model:**
    - This simple model allowed us to quickly gauge initial performance.
    - High overall accuracy (~99.68%) but failed to predict minority class accurately.

    **Addressing Class Imbalance:**
    - **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to generate synthetic samples for the minority class, balancing the dataset.
    
    **Ensemble Modeling and Comparison:**
    - We used ensemble techniques: Random Forest, Decision Tree, and XGBoost.
    - Each model was trained on SMOTE-enhanced, scaled training data.
    - Performance was evaluated using metrics like accuracy, precision, recall, and F1-score.
    """)

    # Model selection
    choice = ['Base Logistic Regression', 'Logistic Regression with SMOTE', 'Decision Tree', 'Random Forest', 'XGBoost']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    # Metric selection
    metric_option = st.radio("Choose a metric to display:", ['Accuracy', 'Classification Report', 'Confusion Matrix'])

    # Results for each model
    if option == 'Base Logistic Regression':
        if metric_option == 'Accuracy':
            st.write("Logistic Regression Accuracy: 0.9968004339590606")
        elif metric_option == 'Classification Report':
            st.text(base_logistic_regression_report)
        elif metric_option == 'Confusion Matrix':
            display_confusion_matrix(base_logistic_regression_confusion_matrix)

    elif option == 'Logistic Regression with SMOTE':
        if metric_option == 'Accuracy':
            st.write("Logistic Regression Accuracy: 0.7730")
        elif metric_option == 'Classification Report':
            st.text(logistic_regression_report)
        elif metric_option == 'Confusion Matrix':
            display_confusion_matrix(logistic_regression_confusion_matrix)

    elif option == 'Decision Tree':
        if metric_option == 'Accuracy':
            st.write("Decision Tree Accuracy: 0.9922")
        elif metric_option == 'Classification Report':
            st.text(decision_tree_report)
        elif metric_option == 'Confusion Matrix':
            display_confusion_matrix(decision_tree_confusion_matrix)

    elif option == 'Random Forest':
        if metric_option == 'Accuracy':
            st.write("Random Forest Accuracy: 0.9924")
        elif metric_option == 'Classification Report':
            st.text(random_forest_report)
        elif metric_option == 'Confusion Matrix':
            display_confusion_matrix(random_forest_confusion_matrix)

    elif option == 'XGBoost':
        if metric_option == 'Accuracy':
            st.write("XGBoost Accuracy: 0.9900")
        elif metric_option == 'Classification Report':
            st.text(xgboost_report)
        elif metric_option == 'Confusion Matrix':
            display_confusion_matrix(xgboost_confusion_matrix)
            
            
    st.header("Modeling with Only Content Features")   
    st.markdown("""
                - The purpose is to integrate nowplaying-RS with spotify dataset for prediction and recommendation.
                - Only audio features were employed.
                - Different over sampling methods were used, random-over sampler, SMOTE and ADASYN; 
                  SMOTE ended with the best result
                - The best model was Random Forest again.
                - However, we couldn't have better results than the previous modeling. 
                """)
    st.image(PCA_PATH)
    if st.button('Show Classification Report'):
       st.text(best_model_content_features) 
            

# Analysis of the Best Model
elif section == "Analysis of the Best Model":
    st.header("Analysis of the Best Model")
    
    st.subheader("Nowplaying Dataset")
    
    st.subheader("Hyperparameter Optimization Using Randomized Search")
    st.markdown("""
    The Random Forest model showed promising results in preliminary tests. To further enhance its performance, RandomizedSearchCV was utilized to optimize several hyperparameters. This approach searches over specified parameter values to find the best combination for improving model performance.
    
    **Parameter Grid:**
    A comprehensive grid of parameters including the number of trees (n_estimators), tree depth (max_depth), minimum samples to split a node (min_samples_split), and minimum samples at a leaf node (min_samples_leaf) was defined.
    
    **Search Process:**
    The search involved fitting various configurations of the Random Forest model on a subset of the training data (2% used for faster computation), using SMOTE for balancing and cross-validation to ensure robustness.
    
    **Best Model Selection:**
    The best performing model parameters (n_estimators: 200, max_depth: 30, min_samples_split: 2, min_samples_leaf: 2, max_features: 'log2) were selected based on cross-validated accuracy scores.
    """)
    
    st.code("""
    Fitting 3 folds for each of 50 candidates, totalling 150 fits
    Best model: RandomForestClassifier(max_depth=30, max_features='log2', min_samples_leaf=2,
                       n_estimators=200, random_state=42)
    Best Parameters: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 30}
    Training Accuracy: 99.8686
    Test Accuracy: 99.4498
    """)
    
    st.subheader("Model Performance")
    metric_option_best_model = st.radio("Choose a metric to display for the best model:", ['Accuracy', 'Classification Report', 'Confusion Matrix'])
    
    if metric_option_best_model == 'Accuracy':
        st.write("Test Accuracy: 99.4498%")
    elif metric_option_best_model == 'Classification Report':
        st.text(best_model_report)
    elif metric_option_best_model == 'Confusion Matrix':
        display_confusion_matrix(best_model_confusion_matrix)
    
    st.subheader("Implementation of Optimized Random Forest Model with Class Weights")
    st.markdown("""
    In this phase of the project, we utilized the best parameters identified from a previous grid search optimization process to configure the Random Forest model. Additionally, instead of relying on SMOTE for balancing the dataset, we explored the use of class weights to manage the imbalance directly within the model training process. This approach allows us to directly influence the model's handling of class imbalance through the learning algorithm itself rather than altering the dataset's composition.
    
    Class weights were manually specified with a greater weight assigned to the minority class (0). The weight ratio (10:1) was chosen to counteract the severe class imbalance by increasing the penalty for misclassifying the minority class, thus focusing the model's learning toward better recognition of these under-represented samples.
    """)
    
    st.subheader("Model Evaluation")
    st.markdown("""
    **Performance Metrics:**
    
    **Accuracy:** The model achieved 99.42% accuracy on the test set.
    
    **Precision, Recall, and F1-Score:** Significant improvement in recall for the minority class (0.92) and an F1-score of 0.50, indicating better identification of the minority class compared to models without class weights.
    
    **Confusion Matrix Analysis:**
    
    Correctly identified 1,330 out of 1,451 instances of the minority class.
    
    2,510 false positives, but a substantial increase in true positives for the minority class.
    """)
    
    metric_option_class_weights = st.radio("Choose a metric to display for the optimized Random Forest model with class weights:", ['Accuracy', 'Classification Report', 'Confusion Matrix'])
    
    if metric_option_class_weights == 'Accuracy':
        st.write("Random Forest Accuracy: 99.42%")
    elif metric_option_class_weights == 'Classification Report':
        st.text(class_weights_report)
    elif metric_option_class_weights == 'Confusion Matrix':
        display_confusion_matrix(class_weights_confusion_matrix)

    st.subheader("Analysis of Feature Importance from Optimized Random Forest Model")
    st.markdown("""
    Following the hyperparameter tuning using RandomizedSearchCV, the optimized Random Forest model provided a detailed view into the relative importance of each feature in predicting music appreciation. The significance of understanding and visualizing feature importance lies in its ability to highlight which features contribute most significantly to the model's decision-making process. This insight is crucial for both refining the model further and for providing actionable insights to stakeholders regarding which aspects of the music or user feedback most influence listener appreciation.
    
    **Plotting Feature Importance:**
    Purpose: The feature importance plot was generated as a part of the model evaluation to visualize and rank the features based on their impact on the model’s predictions. This step is essential for interpreting the model's behavior, particularly in understanding how different features influence the predictive performance.
    
    **Methodology:** Using the feature_importances_ attribute of the trained Random Forest model, we extracted the importance scores for each feature. These scores represent the mean decrease in impurity (usually calculated by the Gini index) contributed by each feature across all trees in the forest.
    """)
    
    # Feature importance plot (replace 'path_to_feature_importance_plot' with the actual path)
    st.image(Feature_importance, caption='Feature Importance for Nowplaying Dataset')

    st.markdown("""
    **Key Observations:**
    - **High Importance Features:** Features such as instrumentalness, loudness, and time-related features like Eastern Time and Pacific Time often appear at the top of the list. High scores in instrumentalness and loudness suggest a strong preference or aversion by listeners towards instrumental music and the loudness of tracks, which directly impacts their appreciation.
    - **Temporal Features:** The importance of time zone features (e.g., Eastern Time, Pacific Time) indicates significant geographical variations in music appreciation, possibly reflecting cultural or regional musical preferences.
    - **Lower Importance Features:** Features appearing towards the bottom of the importance ranking, such as speechiness or valence, might have a lesser direct impact on the prediction of music appreciation in the context of this specific dataset.
    
    **Strategic Implications:**
    - **Model Refinement:** Understanding which features are most influential allows for the optimization of data collection and preprocessing in future iterations of modeling work. For example, focusing on enhancing the quality and range of data related to the most influential features could improve model performance.
    - **Business Insights:** For stakeholders, knowing which features affect music appreciation the most can guide strategic decisions, such as marketing different genres of music in specific regions or adjusting music production elements to align with listener preferences.
    
    In conclusion, the feature importance analysis not only enriches our understanding of the dataset and model but also provides a clear direction for both technical improvements and strategic business actions. This step is indispensable for advancing toward a more targeted and effective music recommendation system.
    """)
    
    
    
    
  
    

# Deep Learning
elif section == "Deep Learning":
    st.header("Deep Learning Model Implementation and Evaluation (Nowplaying Dataset)")

    # Deep Learning Model Architecture
    st.subheader("Deep Learning Model Architecture")
    st.write("""
    The architecture of the deep learning model was designed to capture complex patterns in the data:

    - **Input Layer:** Matches the number of features in the training data.
    - **Hidden Layers:**
        - First Hidden Layer: 128 neurons with ReLU activation.
        - Dropout Layer: 50% dropout rate.
        - Second Hidden Layer: 64 neurons with ReLU activation.
        - Dropout Layer: 50% dropout rate.
        - Third Hidden Layer: 32 neurons with ReLU activation.
        - Dropout Layer: 50% dropout rate.
    - **Output Layer:** Single neuron with sigmoid activation function for binary classification.

    The model uses the Adam optimizer for efficient training and binary crossentropy loss function, ideal for binary classification tasks.
    """)

    # Interpretation of Results
    st.subheader("Interpretation of Results")

    choice = ['Test Accuracy', 'Classification Report', 'Confusion Matrix']
    option = st.selectbox('Choose the metric to display:', choice)

    if option == 'Test Accuracy':
        st.write("**Test accuracy:** 0.9804828763008118")

    elif option == 'Classification Report':
        st.text(deep_learning_report)

    elif option == 'Confusion Matrix':
        display_confusion_matrix(deep_learning_confusion_matrix)

    # Training History
    st.subheader("Training History")
    st.write("""
    The training history plots illustrate the model's performance over epochs:
    
    - **Model Accuracy Plot:** Steady improvement in training and validation accuracy.
    - **Model Loss Plot:** Consistent decrease in training loss, with validation loss generally following the trend.
    """)
    st.image(np_deep, caption="Figure: Training and Validation Accuracy and Loss for the Deep Learning Model")

    # Comparison with Random Forest Model
    st.subheader("Comparison with Random Forest Model")
    st.write("""
    Despite the high test accuracy, the deep learning model's performance was worse than the Random Forest model:
    
    - **Class 0 Precision and Recall:** Deep learning model has significantly lower precision for class 0 compared to the Random Forest model.
    - **Overall Performance:** Random Forest model, with an accuracy of 99.42%, outperformed the deep learning model in precision, recall, and F1-score for both classes.
    """)

    # Reasons for Worse Performance
    st.subheader("Reasons for Worse Performance of Deep Learning Model")
    st.write("""
    - **Imbalanced Data:** The deep learning model struggled with the extreme class imbalance.
    - **Model Complexity:** The complexity of deep learning models requires large amounts of data to generalize well. The imbalance hindered the model's ability to learn meaningful patterns for the minority class.
    
    Overall, while the deep learning model showed promise, the Random Forest model remains the superior choice for this specific task.
    """)

    # Interpretation of the Results
    st.subheader("Interpretation of the Results")
    st.write("""
    **Feature Importance Analysis (Random Forest Model):**
    - Features like 'instrumentalness' and 'loudness' had significant impacts on sentiment prediction.
    - This understanding helped engineer features more effectively and adjust the model parameters to enhance performance.

    **Deep Learning Model:**
    - Despite its architecture capturing complex patterns, it did not outperform the Random Forest model.
    - The high accuracy achieved was marred by issues in precision and recall for the minority class, highlighting potential class imbalance issues not fully mitigated by SMOTE.
    """)


elif section == "Genre Detection":
    st.header("Genre Detection")
    st.markdown(""" The #nowplaying dataset initially only contained IDs for each track and no further identifiers of any kind as to what the name of the song was.
                This meant that the dataset was hard to interpret for a human, which also meant that there would be very little information to rely on in terms of interpretability and performance assessment of the recommendation model.
                Thus, the idea arose to try and assign a genre to each track. This would provide at least a category based on which one can get a rough idea of what a track might sound like. 
    """) 
    st.image(Genre_Example)
    st.subheader("What even is genre?")
    st.markdown("""For the purpose of this project, we made the initial assumption that musical tracks can be put in groups based on shared features or properties; said group can be referred to as genre. 
                Some of these features include the ones that the Spotify API provides: For example, we can assume that tracks belonging to the Dance genre have on average higher danceability scores, and tracks from the Acoustic genre would likely overall have very high acousticness scores. 
                Two caveats to our genre detection efforts have to be noted:  Firstly, neither of our datasets contained information about certain aspects that might be highly influential on genre assignment. Examples of this might be the origin country of the artist or instruments used.
                Secondly, we may not forget that assigning a genre to a piece of music is not an exact science and can be highly subjective, especially when it comes to highly specific labels. We do not know how genre labels were originally assigned in our training set.""")

    st.subheader("Specific Preprocessing")
    st.markdown("""The track_genre column in the Spotify dataset contains 1000 tracks of each of the 114 different genres. At first, attempts were made to train the model on this data frame as is and attempt class prediction with the full set of 114 genres. The results however were less than convincing, the accuracy scores were <0.15. 
                Upon further analysis, we noticed that some of the pre-assigned genre categories were very similar and could therefore not be discerned with accuracy (an example of this would be the labels “electro” and “electronic”). In order to remedy this, we decided to combine the 114 genres into 9 broader genre groups, expecting these would be more distinctive from one another. 
                After assigning the new genre groups, we created a new stratified subset which included 7000 tracks from each of the groups to maintain a balanced dataset. When splitting the data into test and training sets, we also made sure it was stratified.""")

    st.subheader("Training various models") 
    st.markdown("""The first attempts at genre classification were made using “traditional” machine learning methods such as K-Nearest Neighbor, Decision Tree, Random Forest, and Support Vector Machine.
                The parameters for each of these models were chosen via GridSearchCV to optimize for accuracy.

                """)

    model_choice = ['K-nearest Neighbor', 'Decision Tree', 'Random Forest', 'Neural Network']

    option = st.selectbox('Choice of the model', model_choice)
    if option == 'K-nearest Neighbor':
        st.markdown("""Parameters (determined with GridSearch): n_neighbors = 30, metric = "manhattan" """)
        st.subheader("Classification Report")
        st.image("src/streamlit/data/ClassificationReportKNN.PNG")
        st.cache_data.clear()
        #st.image("src/streamlit/data/ConfusionMatrixKNN.png")
        Image.open('src/streamlit/data/ConfusionMatrixKNN.png')
        
    elif option == 'Decision Tree':
        st.markdown("""Parameters (determined with GridSearch): 'criterion': 'gini', 'splitter': 'best' """)
        st.image("src/streamlit/data/ClassificationReportDecTree.PNG")
        st.image("src/streamlit/data/Confusion_matrix_decision_trees.PNG")
        
    elif option == 'Random Forest':
        st.markdown("""Parameters (determined with GridSearch):  'max_depth': 20, 'min_samples_leaf': 1""")
        st.image("src/streamlit/data/ClassificationReportRandomForest.PNG")
        st.image("data/Confusion_matrix_random_forest.PNG")
        
        
    elif option == 'Neural Network':
        st.markdown("""We wanted to see if a deep learning model could potentially outperform the “traditional” approaches and decided to build a sequential dense learning model with the following parameters:  """)
        st.image("src/streamlit/data/Model summary.png")
        st.markdown(""" The classification results were as follows:""")
        st.image("src/streamlit/data/ConfusionMatrixNN.png")
        st.markdown(""" As the performance of the Dense network did not surpass the performance of the Random Forest model, we combined the model with the Random Forest model to create a stacked model, as visualized below: """)
        st.image("src/streamlit/data/ConfusionMatrixNNstacked.png")
        
    
    st.markdown(""" When comparing the classification reports and confusion matrices of the different models, it becomes evident that the models differ not just in their overall accuracy, but also in how well they can classify certain genres over others.""")
    st.markdown(""" We applied the model trained on the Spotify dataset onto the #nowplaying dataset, yielding the following results: """)
    st.image("data/ResultsClassifier.png")

    st.markdown(""" Overall, we must concede that the results of our efforts are less than ideal, but also not a complete failure. The model would have likely performed better if we had additional features available and were not limited to the few features that both datasets shared. Ideally, we would have had audio data, but this would have not been attainable and required handling of an even bigger amount of data and high-end hardware resources, which would have transcended the scope of this project. """)
  
    
elif section == "Recommendation":    
    
    st.write('### Our Recommendation Approach')
    st.markdown("""
                - Spotify API used to add columns to nowplaying-RS
                - track ids used to get the columns below
                - genres are also fetched but discrepancies were observed. 
                """)
    tracks = pd.read_csv(TRACKS_PATH, index_col=0)
    genres = pd.read_csv(GENRES_PATH, index_col=0)
    st.dataframe(tracks.head(5))
    st.write('Genres')
    st.dataframe(genres.head(5))

    st.markdown("""
                &nbsp;
                
                **Content-Based Filtering with Sentiment Prediction**

                &nbsp;

                """)
    resized_schema = resize_image(SCHEMA_PATH, 600) 
    st.image(resized_schema)
  
    data = {
        'User 23247402': {
            'Track': ["I'm Good (Blue)", 'Five Little Ducks'],
            'Artist': ['Kidz Bop Kids', 'Super Simple Songs'],
            'Album': ['KIDZ BOP 2023', 'Baby Shark & More Kids Songs'],
            'Similarity Score': ['0.9999895841731624', '0.9999868876090343']
        },
        'User 863131741': {
            'Track': ['abcdefu', '1553470665499594756'],
            'Artist': ['GAYLE', 'frxgxd'],
            'Album': ['abcdefu', '1553470665499594756'],
            'Similarity Score': ['0.9999950599272903', '0.9999927334549278']
        },
        'User 15518784': {
            'Track': ['Daddy Says No', '414bigfrank (Backpack)'],
            'Artist': ['Haschak Sisters', '414bigfrank'],
            'Album': ['Daddy Says No', '414bigfrank (Backpack)'],
            'Similarity Score': ['0.9999946615642286', '0.9999929018426512']
        },
        'User 17945688': {
            'Track': ['Some Track 1', 'Some Track 2'],
            'Artist': ['Some Artist 1', 'Some Artist 2'],
            'Album': ['KIDZ BOP 2024', '100 Fun Songs for Kids Vol. 1'],
            'Similarity Score': ['0.999993107077451', '0.999993107077451']
        }
    }

    # Function to display data for a specific user
    def display_recommendations(user_id, recommendations):
        st.write(f"Recommendations for User {user_id}:")
        df = pd.DataFrame(recommendations)
        st.table(df)

    # Display recommendations for each user
    for user_id, recommendations in data.items():
        display_recommendations(user_id, recommendations)
  
  
  
# Conclusion
elif section == "Conclusion and Critism":
    st.write("### Conclusion")
    st.markdown("""
                **Random Forest :** 
                - Successfully handles class imbalances with feature engineering and class weighting.
                - Provides interpretable insights via feature importance scores, 
                    crucial for business decisions like predicting song attributes for positive sentiment.

                **Deep Learning:** 
                - Deep Learning model did not surpass the Random Forest model, despite its complexity. 
                - Emphasizes the importance of choosing the right model and managing class imbalances effectively.
                
                **Business Impact:**
                - Insights from the Random Forest model inform strategies for music recommendations, targeted marketing, and product development.
                - Enhances user engagement and satisfaction by understanding key drivers of positive sentiment in music.
                """)
    #The success of the Random Forest model highlights its robustness and ability to handle class imbalance effectively when combined with feature engineering and class weighting techniques. This model's interpretability, through feature importance scores, provides valuable insights for business decisions, such as understanding which song attributes are most predictive of positive sentiment.

    #In contrast, the deep learning model, despite its sophisticated architecture, did not achieve better performance than the Random Forest model. This result underscores the importance of choosing the right model for the problem at hand and ensuring adequate handling of class imbalances. The complexity of deep learning models may not always translate to better performance, especially in scenarios where simpler, well-tuned models like Random Forests can provide more reliable and interpretable results.

    #From a business perspective, the insights derived from the Random Forest model can inform strategies for music recommendation systems, targeted marketing campaigns, and product development focused on enhancing user engagement and satisfaction. Understanding the key drivers of positive sentiment in music can help businesses tailor their offerings to better meet user preferences and improve overall user experience.
    st.write("### Criticism and Outlook")
    st.markdown("""
                - nowplaying-RS data set can be extended with audio or lyrics for better sentiment prediction.
                - More time and more calculation capacity are required to optimize models. 
                - Recommendation approach can be improved with genre analysis. 
                """)
    ##With more time, further optimization of the deep learning model could be explored, including more sophisticated techniques for handling class imbalance. Additionally, integrating more contextual data and experimenting with different model architectures may yield better results. Future work could also involve deploying the model in a live environment and continuously updating it with new data to improve its accuracy and relevance.
    
    


