#Importation des bibliothèques
import os
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.metrics import euclidean_distances
import difflib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")



#PHASE DE CLUSTERING POUR REGROUPER LES GENRES MUSICAUX PROCHES
st.title("Clustering partitionnel à l’aide de l’algorithme K-means")
st.write("Dans cette phase, nous utiliserons des techniques de clustering pour regrouper les genres et les chansons en fonction de leurs caractéristiques. Le clustering permet de découvrir des modèles et des similitudes au sein de l’ensemble de données, ce qui nous permet d’obtenir des informations sur la structure des données musicales.")
genre_data=pd.read_csv('datasets/data_by_genres.csv')
data=pd.read_csv('datasets/data.csv')
#Ensuite, vous définissez un pipeline qui met d’abord à l’échelle les données à l’aide de 10 clusters, puis applique le clustering K-Means avec 10 clusters.
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
#Enfin, vous ajustez le pipeline sur les données numériques et stockez les affectations de cluster dans une nouvelle colonne appelée « cluster » dans le dataframe genre_data
genre_data['cluster'] = cluster_pipeline.predict(X)


#visualiser les clusters que vous avez créés à l’étape précédente à l’aide de t-SNE.
#Définir un nouveau pipeline qui met à l’échelle les données et applique t-SNE pour réduire la dimensionnalité à 2 composants.
#Ajuster et transformer les données numériques à l’aide du pipeline t-SNE pour obtenir un encastrement 2D.X

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
#Les intégrations 2D sont stockées dans un DataFrame appelé , avec les genres et les affectations de cluster correspondants à partir du DataFrame.projectiongenre_data

projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']
st.subheader("Visualisation avec TNSE")
fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
st.plotly_chart(fig, use_container_width=True)


song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20,
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels




# Visualisation des clusters avec PCA

st.subheader("Visualisation avec PCA")
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
st.plotly_chart(fig, use_container_width=True)





### PHASE DE MODELISATION DE NOTRE SYSTEME DE RECOMMANDATION SPOTIFY

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="df9e083676db412897123a91de04a341",
                                                           client_secret="f4d011d584a7447a83573544f1e559bd"))

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):

    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):

    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict

def recommend_songs( song_list, spotify_data, n_songs=10):

    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')




with st.form('Spotify'):
    #a = st.text_input('Chanson 1 ')
    #b = st.number_input('Date de sortie 1')
    #a1 = st.text_input('Chanson 2')
    #b1 = st.number_input('Date de sortie 2')
    #submit = st.form_submit_button('Que me recommandez-vous ?')

if submit:
    list_songs=[{'name': a, 'year':b},
                #{'name': a1, 'year':b1}]
    lst=recommend_songs(list_songs, data)
    s = ''
    for i in lst:
        #st.write(i)



