import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import json

from projet_rail_estate.double_ml_logic.load_data import load_dvf_gare
from projet_rail_estate.double_ml_logic.clean_data import clean_data_dvf, clean_data_new_gares


## --- Configuration de l'application ---
st.set_page_config(
    page_title="Rail Estate",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Rail Estate")

API_BASE_URL = "http://127.0.0.1:8000/api/data" # Exemple d'URL locale

def fetch_map_data():
    """Simule l'appel à l'API pour récupérer les données de la carte."""
    # Simuler un appel API réussi
    # response = requests.get(f"{API_BASE_URL}/map")
    # if response.status_code == 200:
    #     data = response.json()
    #     return pd.DataFrame(data)

    st.info("Simule l'appel à la route API `/map`...")

    # Création de données aléatoires pour simuler des points en Île-de-France
    data = {
        'lat': 48.8660 + np.random.randn(100) * 0.1,  # Latitude autour de Paris
        'lon': 2.3550 + np.random.randn(100) * 0.1,  # Longitude autour de Paris
        'price': np.random.randint(150000, 500000, 100),
    }
    return pd.DataFrame(data)

@st.cache_data
def fetch_bar_plot_data():
    """Simule l'appel à l'API pour récupérer les données des bar plots détaillées."""
    # st.info("Simule l'appel à la route API `/bars`...")

    # Données simulées incluant Ville, Gare/Station et Ligne de Métro
    data = {
        'Ville': np.random.choice(['Paris 12e', 'Vincennes', 'Montreuil', 'Saint-Denis', 'Versailles'], 100),
        'Gare_Station': np.random.choice(['Gare de Lyon', 'Nation', 'Château de Vincennes', 'Saint-Denis Pleyel', 'RER A'], 100),
        'Ligne_Metro': np.random.choice(['L1', 'L4', 'RER A', 'RER D', 'L9', 'Pas de Métro'], 100),
        'Prix_Moyen': np.random.randint(4000, 15000, 100) # Prix moyen au m² simulé
    }
    return pd.DataFrame(data)

# ... (fetch_prediction_data inchangée) ...

@st.cache_data
def fetch_prediction_data():
    """Simule l'appel à l'API pour récupérer les données de prédiction."""
    # Simuler un appel API réussi
    # response = requests.get(f"{API_BASE_URL}/prediction")
    # if response.status_code == 200:
    #     data = response.json()
    #     return pd.DataFrame(data)

    st.info("Simule l'appel à la route API `/prediction`...")

    # Création de données aléatoires pour la comparaison Réel vs Prédit
    data = {
        'Mois': pd.to_datetime(['2025-01', '2025-02', '2025-03', '2025-04', '2025-05']),
        'Réel': [300, 320, 280, 350, 330],
        'Prédit': [290, 315, 290, 360, 340],
    }
    return pd.DataFrame(data).set_index('Mois')

# ==============================================================================
# 1. Carte de l'Île-de-France
# ==============================================================================

# st.header("Localisation des Biens en Île-de-France")
# st.markdown("Cette carte affichera les biens immobiliers récupérés via l'API, montrant la distribution géographique des annonces.")

# map_data = fetch_map_data()

# if not map_data.empty:
#     st.subheader("Distribution des Biens")
#     # Affiche la carte en utilisant les colonnes 'lat' et 'lon'
#     st.map(map_data, latitude='lat', longitude='lon', zoom=9)
#     # Afficher un extrait des données pour vérification
#     with st.expander("Voir les données brutes de la carte"):
#         st.dataframe(map_data.head())
# else:
#     st.error("Impossible de récupérer les données de la carte depuis l'API.")


# ------------------------------------------------------------
# CONFIG STREAMLIT
# ------------------------------------------------------------

st.set_page_config(
    page_title="Prix m² IDF – DVF & Gares GPE",
    layout="wide"
)

st.title("📍 Évolution du prix moyen au m² en Île-de-France (2014–2025)")
st.write("Visualisation DVF + Polygones Communes + Nouvelles gares GPE")


# ------------------------------------------------------------
# LOAD DATA (CACHÉ)
# ------------------------------------------------------------
@st.cache_data
def load_communes():
    url = (
        "https://geo.api.gouv.fr/communes"
        "?codeRegion=11&format=geojson&geometry=contour"
    )
    gdf = gpd.read_file(url)
    gdf = gdf.rename(columns={"code": "code_commune", "nom": "commune"})
    gdf["code_insee"] = gdf["code_commune"].astype(str).str.zfill(5)

    return gdf[["code_insee", "commune", "geometry"]]


@st.cache_data
def load_all_data():
    df_dvf, df_gares = load_dvf_gare(
        bucket_name="rail-estate-data",
        uri_dvf="dvf/dvf_idf_2014_2025.csv",
        uri_gare="gares/csv_gares.csv"
    )
    return clean_data_dvf(df_dvf), clean_data_new_gares(df_gares)


gdf_communes = load_communes()
df_dvf_clean, df_new_gares_clean = load_all_data()

geojson_communes = json.loads(gdf_communes.to_json())


# ------------------------------------------------------------
# DVF AGGREGATION
# ------------------------------------------------------------
dvf = df_dvf_clean.copy()
dvf["code_insee"] = dvf["Code ville"].astype(str).str.zfill(5)
dvf = dvf.query("2014 <= annee <= 2025")

agg_year = (
    dvf.groupby(["code_insee", "Commune", "annee"], as_index=False)
    .agg(
        prix_m2_moyen=("prix_m2", "mean"),
        nb_tx_annee=("prix_m2", "size"),
    )
)

agg_period = (
    dvf.groupby(["code_insee", "Commune"], as_index=False)
    .agg(
        prix_m2_moyen_periode=("prix_m2", "mean"),
        nb_tx_periode=("prix_m2", "size"),
    )
)

nb_years = dvf["annee"].nunique()
agg_period["nb_tx_moyen_par_an"] = agg_period["nb_tx_periode"] / nb_years

df_choro = agg_year.merge(
    agg_period, on=["code_insee", "Commune"], how="left"
).rename(columns={"Commune": "commune"})

df_choro = df_choro[df_choro["code_insee"].isin(gdf_communes["code_insee"])]

vmin = df_choro["prix_m2_moyen"].quantile(0.05)
vmax = df_choro["prix_m2_moyen"].quantile(0.95)


# ------------------------------------------------------------
# SIDEBAR – ANNEE & OPTIONS
# ------------------------------------------------------------
st.sidebar.header("⚙️ Paramètres")

annee_select = st.sidebar.slider(
    "Sélectionne une année",
    min_value=2014,
    max_value=2025,
    value=2020
)

show_gares = st.sidebar.checkbox("Afficher les nouvelles gares GPE", value=True)


# ------------------------------------------------------------
# FILTRE ANNEE
# ------------------------------------------------------------
df_year = df_choro[df_choro["annee"] == annee_select]


# ------------------------------------------------------------
# MAP PLOTLY
# ------------------------------------------------------------
fig = px.choropleth_mapbox(
    df_year,
    geojson=geojson_communes,
    featureidkey="properties.code_insee",
    locations="code_insee",
    color="prix_m2_moyen",
    hover_name="commune",
    hover_data={
        "annee": True,
        "prix_m2_moyen": ":.0f",
        "nb_tx_annee": True,
    },
    color_continuous_scale="RdYlGn_r",
    range_color=(vmin, vmax),
    mapbox_style="carto-positron",
    zoom=9,
    center={"lat": 48.8566, "lon": 2.35},
    opacity=0.7,
)

# ------------------------------------------------------------
# AJOUT DES GARES GPE
# ------------------------------------------------------------
if show_gares:
    new_gares = df_new_gares_clean.dropna(subset=["latitude", "longitude"])
    fig.add_trace(
        go.Scattermapbox(
            lat=new_gares["latitude"],
            lon=new_gares["longitude"],
            mode="markers",
            marker=dict(size=7, color="blue"),
            text=new_gares["nom_gare"],
            hovertemplate="<b>%{text}</b><extra>Nouvelle gare GPE</extra>",
            name="Gares GPE"
        )
    )

fig.update_layout(
    title=f"Prix moyen au m² – {annee_select}",
    height=800,
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
)

st.plotly_chart(fig, use_container_width=True)














# ==============================================================================
# 2. Bar Plots : Prix Moyens selon le Filtre
# (Mise à jour pour utiliser st.selectbox)
# ==============================================================================

st.header("📊 2. Prix Moyen selon le Filtre (Bar Plots)")
st.markdown("Utilisez le menu déroulant ci-dessous pour agréger le prix moyen par ville, gare/station ou ligne de métro, obtenues via l'API.")

bar_data = fetch_bar_plot_data() # Utilise la fonction de simulation mise à jour

if not bar_data.empty:

    # ----------------------------------------------------
    # WIDGET DE SÉLECTION (st.selectbox)
    # ----------------------------------------------------

    # Options de sélection (vous pouvez personnaliser les libellés ici)
    options_filtre = {
        'Ville': 'Ville',
        'Gare / Station': 'Gare_Station',
        'Ligne de Métro / RER': 'Ligne_Metro'
    }

    # Création du selectbox (dropdown)
    filtre_libelle = st.selectbox(
        "Sélectionnez le critère d'agrégation :",
        options=list(options_filtre.keys()),
        index=0 # Ville par défaut
    )

    # Définition de la colonne de données correspondante
    colonne_filtre = options_filtre[filtre_libelle]

    # Agrégation des données
    df_agg = bar_data.groupby(colonne_filtre)['Prix_Moyen'].mean().reset_index()
    df_agg.columns = [colonne_filtre, 'Prix_Moyen_Agg']

    # Trier par prix moyen agrégé
    df_agg = df_agg.sort_values(by='Prix_Moyen_Agg', ascending=False)

    # ----------------------------------------------------
    # AFFICHAGE DU GRAPHIQUE
    # ----------------------------------------------------

    st.subheader(f"Prix Moyen au m² agrégé par : {filtre_libelle}")

    import plotly.express as px

    fig_bar = px.bar(
        df_agg,
        x='Prix_Moyen_Agg',
        y=colonne_filtre,
        orientation='h',
        color='Prix_Moyen_Agg',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'Prix_Moyen_Agg': 'Prix Moyen au m² (€)', colonne_filtre: filtre_libelle}
    )

    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander(f"Voir les données agrégées par {filtre_libelle}"):
        st.dataframe(df_agg)

else:
    st.error("Impossible de récupérer les données des bar plots depuis l'API.")



# ... (Section 3 inchangée) ...

# ... (Section 3 inchangée) ...

# ==============================================================================
# 3. Graphique Réel vs Prédit
# ==============================================================================

st.header("Évolution Future Réel vs Prédit")
st.markdown("Ce graphique montrera la performance de votre modèle de prédiction (Réel) comparée aux valeurs effectivement observées (Prédit).")

prediction_data = fetch_prediction_data()

if not prediction_data.empty:
    st.subheader("Comparaison Temporelle (Nombre de Transactions)")

    # Créer un line chart avec Streamlit (ou Plotly/Matplotlib pour plus de contrôle)
    st.line_chart(prediction_data, use_container_width=True)

    with st.expander("Voir les données brutes de prédiction"):
        st.dataframe(prediction_data)
else:
    st.error("Impossible de récupérer les données de prédiction depuis l'API.")
