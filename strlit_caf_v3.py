import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from shapely.geometry import shape
import json
import joblib
import os
import prophet

# Configuration de la page pour occuper toute la largeur centrale
st.set_page_config(page_title="Prédiction des dépenses CAF département", layout="wide")

chemin = "datasets/datasets_destination/"

# Chargement des datasets
@st.cache_data
def load_data():
    df_chomage = pd.read_csv(f"{chemin}df_chomage_BIT_france_dep.csv")
    df_chomage["Code"] = df_chomage["Code"].replace({"2A": "201", "2B": "202"}).astype(int)
    df_pop = pd.read_csv(f"{chemin}df_population_france_dep_ml_prophet_2025_2026.csv")
    df_naiss = pd.read_csv(f"{chemin}df_naissances_annuelles_france_dep_ml_prophet_2025_2026.csv")

    with open("departements-avec-outre-mer.geojson") as f:
        geojson = json.load(f)
    return df_chomage, df_pop, df_naiss, geojson

df_chomage, df_pop, df_naiss, geojson = load_data()

# Liste des départements (DOM en bas)
dom_codes = ["971", "972", "973", "974"]

def safe_code_sort_key(code):
    try:
        return int(code)
    except ValueError:
        return float("inf")

features_metropole = [
    f for f in geojson["features"]
    if f["properties"]["code"] not in dom_codes + ["976"]
]
features_dom = [
    f for f in geojson["features"]
    if f["properties"]["code"] in dom_codes
]

dep_metropole = {
    f["properties"]["nom"]: f["properties"]["code"]
    for f in sorted(features_metropole, key=lambda f: safe_code_sort_key(f["properties"]["code"]))
}
dep_dom = {
    f["properties"]["nom"]: f["properties"]["code"]
    for f in sorted(features_dom, key=lambda f: f["properties"]["code"])
}

departements = {**dep_metropole, **dep_dom}

# Sidebar UI
st.sidebar.title("Configuration")
selected_dep = st.sidebar.selectbox("Département", list(departements.keys()))
selected_year = st.sidebar.selectbox("Année", [2024, 2025, 2026])
if selected_year == 2024:
    month_options = [9, 10, 11, 12]
elif selected_year == 2026:
    month_options = list(range(1, 9))
else:
    month_options = list(range(1, 13))
selected_month = st.sidebar.selectbox("Mois", month_options)
target_options = {
    "Toutes prestations (NDUR)": "indmtt_ndur",
    "Prestations petites enfances (NDURPAJE)": "indmtt_ndurpaje",
    "Prestation prime de naissance (PN)": "indmtt_pn",
    "Prestations enfances (NDUREJ)": "indmtt_ndurej"
}
selected_target_label = st.sidebar.selectbox("Type de dépense à prédire", list(target_options.keys()))
selected_target = target_options[selected_target_label]

# Titre principal
st.title("Prédiction des dépenses CAF département")

dep_code = departements[selected_dep]
dep_code_int = int(dep_code) if dep_code.isdigit() else None  # pour matcher les CSV uniquement si numérique

# Calcul du centre du département sélectionné pour centrage de la carte
map_location = [46.6, 1.8]  # par défaut : centré sur la France métropolitaine
zoom = 6

for feature in geojson["features"]:
    if feature["properties"]["code"] == dep_code:
        coords = feature["geometry"]["coordinates"]
        if feature["geometry"]["type"] == "Polygon":
            points = coords[0]
        elif feature["geometry"]["type"] == "MultiPolygon":
            points = coords[0][0]
        else:
            continue
        lats = [pt[1] for pt in points]
        lons = [pt[0] for pt in points]
        map_location = [sum(lats) / len(lats), sum(lons) / len(lons)]
        break
else:
    map_location = [46.6, 1.8]
    zoom = 6

# Création de la carte
m = folium.Map(location=map_location, zoom_start=8)

folium.GeoJson(
    geojson,
    name="Départements",
    style_function=lambda x: {"fillColor": "yellow" if x["properties"]["code"] == dep_code else "#eeeeee",
                            "color": "black", "weight": 1},
    tooltip=folium.GeoJsonTooltip(fields=["nom"], aliases=["Département:"])
).add_to(m)

# Ajouter un point sur le département sélectionné
for feature in geojson["features"]:
    if feature["properties"]["code"] == dep_code:
        geom = shape(feature["geometry"])
        centroid = geom.centroid
        folium.Marker(location=[centroid.y, centroid.x], popup=selected_dep).add_to(m)
        break

# Création des colonnes : la carte à gauche, les prédictions à droite
col_map, col_pred = st.columns(2)

with col_map:
    st.subheader("Carte du département")
    folium_static(m)

# Préparation des données pour les prédictions
def mois_vers_trimestre(mois):
    if mois in [1, 2, 3]:
        return "T1"
    elif mois in [4, 5, 6]:
        return "T2"
    elif mois in [7, 8, 9]:
        return "T3"
    else:
        return "T4"

trimestre = mois_vers_trimestre(selected_month)
year_n2 = selected_year - 2
trimestre_str = f"{trimestre}_{year_n2}"

# Taux de chômage N-2
chomage_row = df_chomage[
    (df_chomage["Code"].astype(str).str.zfill(2) == str(dep_code).zfill(2)) &
    (df_chomage["Trimestre"] == trimestre_str)
]
taux_chomage = chomage_row["OBS_VALUE"].values[0] if not chomage_row.empty else None

# Population
if dep_code_int is not None:
    pop_row = df_pop[
        (df_pop["code_dep_population"] == dep_code_int) & 
        (df_pop["population_annee"] == selected_year)
    ]
else:
    pop_row = pd.DataFrame()
population = pop_row["population"].values[0] if not pop_row.empty else None

# Naissances
if dep_code_int is not None:
    naiss_row = df_naiss[
        (df_naiss["code_dep_naissances"] == dep_code_int) & 
        (df_naiss["nb_naissances_annee"] == selected_year)
    ]
else:
    naiss_row = pd.DataFrame()
nb_naissances = naiss_row["nb_naissances"].values[0] if not naiss_row.empty else None

with col_pred:
    st.subheader("Données utilisées pour la prédiction")
    st.write(f"**Taux de chômage ({trimestre_str}) :** {taux_chomage if taux_chomage is not None else 'Non disponible'}")
    st.write(f"**Population ({selected_year}) :** {population if population is not None else 'Non disponible'}")
    st.write(f"**Nombre de naissances ({selected_year}) :** {nb_naissances if nb_naissances is not None else 'Non disponible'}")

    # Prédiction si toutes les données sont présentes
    if None in (taux_chomage, population, nb_naissances):
        st.warning("Données manquantes pour ce département ou cette période.")
    else:
        # Définir le chemin du modèle en fonction du département sélectionné
        dep_str = str(dep_code).zfill(2)
        model_path = os.path.join("ml_prophet", selected_target, f"prophet_model_{dep_str}.pkl")
        model = joblib.load(model_path)

        # Création de la date pour la prédiction à partir de l'année et du mois sélectionnés
        future_date = pd.to_datetime(f"{selected_year}-{selected_month:02d}-01")
        
        # Préparer le DataFrame pour la prédiction
        input_data = pd.DataFrame({
            "ds": [future_date],
            "taux_chomage": [taux_chomage],
            "population": [population],
            "nb_naissances": [nb_naissances]
        })
        
        # Faire la prédiction (la colonne 'yhat' contient la prévision)
        forecast = model.predict(input_data)
        prediction = forecast["yhat"].values[0]
        
        st.success(f"📊 Prédiction ({selected_target_label}) pour {selected_dep} — {selected_month:02d}/{selected_year} : **{prediction:,.2f} €**"
)

