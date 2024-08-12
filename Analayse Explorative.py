
import os
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt

data_year=pd.read_csv('datasets/data_by_year.csv')
data = pd.read_csv('datasets/data.csv')

st.title("Analayse Explorative ")
st.write("Au cours de cette phase, vous effectuerez une analyse exploratoire des données sur l’ensemble de données Spotify. Cette analyse vous aidera à comprendre les tendances des caractéristiques sonores au fil des décennies, à examiner la popularité et les caractéristiques des principaux genres et artistes, et à générer des nuages de mots pour visualiser les genres et les artistes présents dans l’ensemble de données")
st.header("Visualisations")
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(data_year, x='year', y=sound_features, title="Tendances des caractéristiques sonores au fil des décennies")
st.plotly_chart(fig, use_container_width=True)
st.success("L'évolution des caractéristiques sonores au fil des décennies montre une baisse notoire de la musiqe acoustique à voix instrumentale, ainsi qu'une popularisation des musiques énergétiques adaptés à la danse. ")
st.subheader('Les genres les plus populaires en 2020')
st.image("images/genrespopulaires.png")

st.success("Cette visualisation appuie les résultas du premier graphe, en témoignant du changement de goûts musicaux des mélomanes au fil des générations. L'afro soul, l'afro swing, le bashall(aya nakamura), l'alberta hip hop et le chinese electropop, occupent le top 5 des genres musicaux sur spotify en 2020")
st.image("images/topartists.png")


