import os
#BASE_DIR_CANSU = 'C:/Users/Cometaca/Streamlit/'
#BASE_GITHUB = 'src/streamlit/'

#BASE_DIR = 'C:/Users/Cometaca/Streamlit/'
#data_dir = os.path.join(BASE_DIR, 'data')
data_dir = os.path.join(os.path.dirname(__file__), 'data')


#DATASETS
SPOTIFY_DATASET_PATH = os.path.join(data_dir, 'spotify_dataset.csv')
Nowplaying_DATASET_PATH = os.path.join(data_dir, 'module4_cleaned.csv')
TRACKS_PATH = os.path.join(data_dir, 'tracks.csv')
IMAGE_PATH = os.path.join(data_dir, 'music_5.jpg' )
GENRES_PATH = os.path.join(data_dir, 'df_genres_5000.csv')
SCHEMA_PATH = os.path.join(data_dir, 'schema_recommendation.png')
PCA_PATH = os.path.join(data_dir, 'PCA.png')
Sentiment_dis= os.path.join(data_dir, 'Sentiment_count.png')
np_features_dis= os.path.join(data_dir, 'continuous features.png')
np_corr= os.path.join(data_dir, 'np_correlation_matrix.png')
Feature_importance= os.path.join(data_dir, 'Feature importance_main.png')
np_deep= os.path.join(data_dir, 'deep_learning_model loss_model acurracy.png')
Track_per_genre = os.path.join(data_dir, 'track_genre_pop.png')
Spotify_dis = os.path.join(data_dir, 'Spotify_dis.png')
Spotify_cor = os.path.join(data_dir, 'spotify_correlation.png')
