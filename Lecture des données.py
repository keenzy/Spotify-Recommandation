
import os
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

st.title('LECTURE DES DONNEES')


st.write("Dans cette phase, l’objectif est de lire l’ensemble de données Spotify et d’explorer son contenu. Cette étape est cruciale pour comprendre la structure des données et la préparer aux phases ultérieures du projet.")
st.header("Importation du dataset")
st.warning("data = pd.read_csv('datasets/data.csv')")
data_year=pd.read_csv('datasets/data_by_year.csv')
data = pd.read_csv('datasets/data.csv')
st.write(data.head())
st.header("Nettoyage des données")
st.subheader("verification des valeurs nulles ou dupliquées")
st.warning("data.isnull().sum()")
st.success(data.isnull().sum())
st.warning("data.duplicated().sum()")
st.success(data.duplicated().sum())


fig1 = px.box(data, y="valence")
fig2 = px.box(data, y="energy")
fig3 = px.box(data, y="acousticness")
fig4 = px.box(data, y="danceability")
fig5 = px.box(data, y="tempo")

fig = make_subplots(rows=2, cols=4, subplot_titles=['Valence', 'Energy','Acoustic','Danceabilility','tempo'])

fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.add_trace(fig3.data[0], row=1, col=3)
fig.add_trace(fig4.data[0], row=1, col=4)
fig.add_trace(fig5.data[0], row=2, col=1)


st.plotly_chart(fig, use_container_width=True)