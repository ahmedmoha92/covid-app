import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import joblib
from datetime import datetime
import plotly.express as px

# === Configuration de la page ===
st.set_page_config(
    page_title="Dashboard Épidémique COVID",
    layout="wide",
    page_icon="🦠"
)

# === Fonctions de chargement avec cache ===
@st.cache_data
def load_models():
    """Charge tous les modèles et le scaler"""
    return {
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl"),
        "Linear Regression": joblib.load("linear_regression_model.pkl"),
        "Neural Network": joblib.load("mlp_model.pkl"),
        "SVM": joblib.load("svm_model.pkl")
    }, joblib.load("scaler.pkl")

@st.cache_data
def load_data():
    """Charge les données principales"""
    df = pd.read_csv("dataset.csv", encoding="ISO-8859-1")
    gdf = gpd.read_file("regions.geojson")
    return df, gdf

# === Chargement initial ===
with st.spinner("Chargement des données et modèles..."):
    models, scaler = load_models()
    df, gdf = load_data()

# === Constantes ===
CODE_TO_REGION = {
    11: "Île-de-France", 24: "Centre-Val de Loire", 27: "Bourgogne-Franche-Comté",
    28: "Normandie", 32: "Hauts-de-France", 44: "Grand Est", 52: "Pays de la Loire",
    53: "Bretagne", 75: "Nouvelle-Aquitaine", 76: "Occitanie",
    84: "Auvergne-Rhône-Alpes", 93: "Provence-Alpes-Côte d'Azur", 94: "Corse"
}

FEATURES = [
    'retail_and_recreation', 'grocery_and_pharmacy', 'parks',
    'transit_stations', 'workplaces', 'residential',
    'TMin (°C)', 'TMax (°C)', 'TMoy (°C)',
    'pm25', 'pm10', 'o3', 'no2', 'Vaccination'
]
TARGET = 'Cas epidemiques'

# === Nettoyage et Préparation des données ===
def clean_data(df):
    """Nettoie et prépare le dataframe sans supprimer les lignes incomplètes"""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Conversion des dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

    # Conversion des valeurs numériques avec conservation des NaN
    for col in FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

with st.spinner("Préparation des données..."):
    df = clean_data(df)
    df['Region_Name'] = df['Code Insee Region'].map(CODE_TO_REGION)
    df.dropna(subset=['date', 'Region_Name'], inplace=True) # Assurer qu'il n'y a pas de NaN après le mapping

    # Pour l'analyse temporelle, on garde toutes les dates mais on filtre les années
    df = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]

# === Dernières données pour prédiction ===
latest = df.sort_values("date").groupby("Region_Name").tail(1)
X_latest = latest[FEATURES]
X_scaled = scaler.transform(X_latest)

# === Barre latérale ===
st.sidebar.title("⚙️ Paramètres du modèle")
selected_model_name = st.sidebar.selectbox(
    "Choisissez un modèle de prédiction",
    list(models.keys())
)
# Instructions pour exécuter l'application Streamlit
st.sidebar.markdown("""
---
**Pour exécuter cette application :**
1.  Enregistrez ce code sous `app.py`.
2.  Assurez-vous d'avoir `random_forest_model.pkl`, `xgboost_model.pkl`,
    `linear_regression_model.pkl`,
    `mlp_model.pkl`,
    `svm_model.pkl`, `scaler.pkl`, `regions.geojson` et `Dataset2.csv`
    dans le même répertoire.
3.  Ouvrez votre terminal et naviguez vers ce répertoire.
4.  Exécutez la commande : `streamlit run app.py`
""")
model = models[selected_model_name]

# Prédiction
try:
    latest["prediction"] = model.predict(X_scaled)
    # Rendre les prédictions négatives nulles (non-sens physique)
    latest["prediction"] = latest["prediction"].apply(lambda x: max(0, x))
except Exception as e:
    st.error(f"Erreur lors de la prédiction : {str(e)}")
    st.stop()

# === Interface principale ===
st.title("🦠 Tableau de Bord Épidémique COVID-19 - France")

tabs = st.tabs([
    "📘 Contexte",
    "🗺️ Carte et Alertes",
    "📈 Évolution temporelle",
    "🔍 Simulation personnalisée"
])

# === Onglet 1 : Contexte ===
with tabs[0]:
    st.header("📘 Contexte et Objectif")
    st.markdown("""
    Ce tableau de bord vise à anticiper les pics épidémiques de COVID-19 à l'échelle régionale.
    Il s'appuie sur des modèles de machine learning entraînés à partir de données :
    - de mobilité (Google),
    - de pollution (WAQI),
    - météo (Météo France),
    - de couverture vaccinale (data.gouv.fr).

    **Objectif** : Fournir un outil d'aide à la décision pour les Agences Régionales de Santé (ARS) afin de cibler rapidement les zones à risque et d'allouer les ressources de manière proactive.
    """)

    st.subheader("🔧 Modèles disponibles")
    st.write(f"Modèle actuellement sélectionné : **{selected_model_name}**")
    st.json({model_name: type(model_obj).__name__ for model_name, model_obj in models.items()})

    st.subheader("📊 Aperçu des données utilisées")
    st.dataframe(df.head(), hide_index=True)

# === Onglet 2 : Carte interactive ===
with tabs[1]:
    st.header("🗺️ Carte des prédictions épidémiques par région")

    col1, col2 = st.columns([1, 3])
    with col1:
        seuil_alerte = st.slider(
            "Seuil d'alerte épidémique",
            0, 10000, 3000, 100,
            help="Définissez le nombre de cas hebdomadaires prédits à partir duquel une alerte est déclenchée."
        )

        # Ajout des métadonnées
        latest["Statut"] = latest["prediction"].apply(
            lambda x: ("🔴 Alerte" if x > seuil_alerte else
                      "🟡 Modéré" if x > seuil_alerte * 0.7 else
                      "🟢 Stable")
        )

        # Résumé des alertes
        st.metric(
            "Régions en alerte",
            f"{sum(latest['prediction'] > seuil_alerte)} / {len(latest)}"
        )

    with col2:
        # Carte Folium
        m = folium.Map(location=[46.5, 2.5], zoom_start=5.5, tiles="cartodb positron")

        # Fusionner les prédictions avec les données géo
        map_data = gdf.merge(latest, left_on='nom', right_on='Region_Name')

        for _, row in map_data.iterrows():
            pred = row["prediction"]
            region_nom = row["Region_Name"]
            statut = row["Statut"]
            
            color = "red" if statut == "🔴 Alerte" else "orange" if statut == "🟡 Modéré" else "green"
            
            centroid = row.geometry.centroid
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=8 + (pred / 400),  # Taille proportionnelle aux cas
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(f"""
                    <b>{region_nom}</b><br>
                    Cas prédits : {int(pred):,}<br>
                    Statut : {statut}<br>
                    Date des données : {row['date'].strftime('%d/%m/%Y')}
                """, max_width=250)
            ).add_to(m)

        st_folium(m, width=800, height=500, returned_objects=[])

    # Légende et données
    st.markdown("""
    ### 🧭 Légende des statuts :
    - 🟢 **Stable** : prédictions inférieures à 70% du seuil
    - 🟡 **Modéré** : prédictions entre 70% et 100% du seuil
    - 🔴 **Alerte** : prédictions dépassant le seuil
    """)

    # Tableau récapitulatif
    st.subheader("📋 Résumé par région")
    summary_df = latest[[
        "Region_Name", "prediction", "Statut", "date"
    ]].rename(columns={
        "Region_Name": "Région",
        "prediction": "Cas Prédits",
        "date": "Date des données"
    }).sort_values("Cas Prédits", ascending=False)

    st.dataframe(
        summary_df.style.format({
            "Cas Prédits": "{:,.0f}",
            "Date des données": lambda x: x.strftime("%d/%m/%Y")
        }),
        use_container_width=True,
        hide_index=True
    )

    # Export
    csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger les prédictions (CSV)",
        csv,
        f"predictions_covid_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

# === Onglet 3 : Évolution temporelle ===
with tabs[2]:
    st.header("📈 Évolution temporelle des cas par région")

    # Sélection des régions
    selected_regions = st.multiselect(
        "Sélectionner les régions à afficher",
        options=sorted(list(CODE_TO_REGION.values())),
        default=list(CODE_TO_REGION.values())[:5] # Par défaut, les 5 premières régions
    )

    if selected_regions:
        df_filtered = df[df['Region_Name'].isin(selected_regions)]

        st.subheader("Évolution par région (cas observés)")
        fig = px.line(
            df_filtered,
            x='date',
            y=TARGET,
            color='Region_Name',
            title="Évolution des cas épidémiques observés par région",
            labels={'date': 'Date', TARGET: 'Nombre de cas hebdomadaires'}
        )
        fig.update_layout(height=500, legend_title_text='Région')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Vue agrégée - Contribution de chaque région")
        fig2 = px.area(
            df_filtered,
            x='date',
            y=TARGET,
            color='Region_Name',
            title="Contribution de chaque région aux cas totaux (parmi la sélection)"
        )
        fig2.update_layout(height=500, legend_title_text='Région')
        # *** CORRECTION : Affichage du graphique fig2 qui était manquant ***
        st.plotly_chart(fig2, use_container_width=True)
    else:
        # *** AMÉLIORATION : Message si aucune région n'est sélectionnée ***
        st.info("Veuillez sélectionner au moins une région pour afficher les graphiques.")


# === Onglet 4 : Simulation personnalisée ===
with tabs[3]:
    st.header("🔍 Simulation personnalisée")
    st.write("Modifiez les paramètres ci-dessous pour simuler l'impact sur le nombre de cas épidémiques, selon le modèle sélectionné.")

    col_guide, col_simul = st.columns([1, 2])

    with col_guide:
        st.subheader("📚 Guide des variables")
        with st.expander("🚶 Mobilité Humaine", expanded=True):
            st.markdown("""
            **Variation (%) par rapport à une période de référence pré-pandémique.**
            - **🛍️ Commerces/loisirs**: Restaurants, cinémas, etc.
            - **🥦 Épiceries/pharmacies**: Magasins essentiels.
            - **🌳 Parcs**: Espaces verts publics.
            - **🚆 Transports**: Gares, métros, bus.
            - **🏢 Lieux de travail**: Bureaux.
            - **🏠 Résidences**: Temps passé au domicile.
            """)
        with st.expander("🌍 Environnement"):
            st.markdown("""
            **Conditions météorologiques et de pollution moyennes.**
            - **Températures**: Min, Max, Moyenne (°C).
            - **Polluants**: PM2.5, PM10, NO2, O3 (µg/m³).
            """)
        with st.expander("💉 Santé Publique"):
            st.markdown("""
            **Couverture vaccinale**
            - Nombre total de doses administrées dans la région.
            """)

    with col_simul:
        st.subheader("📊 Paramètres de simulation")

        with st.form("simulation_form"):
            st.markdown("#### 🚶 Mobilité")
            mob_cols = st.columns(2)
            retail = mob_cols[0].number_input("Commerces/loisirs (%)", -100, 100, 0)
            grocery = mob_cols[1].number_input("Épiceries/pharmacies (%)", -100, 100, 0)
            parks = mob_cols[0].number_input("Parcs (%)", -100, 100, 0)
            transit = mob_cols[1].number_input("Transports (%)", -100, 100, -10)
            workplaces = mob_cols[0].number_input("Lieux de travail (%)", -100, 100, -15)
            residential = mob_cols[1].number_input("Résidences (%)", -100, 100, 5)

            st.markdown("#### 🌍 Environnement")
            env_cols = st.columns(2)
            tmin = env_cols[0].number_input("Temp. minimale (°C)", -20.0, 40.0, 10.0, format="%.1f")
            tmax = env_cols[1].number_input("Temp. maximale (°C)", -10.0, 50.0, 20.0, format="%.1f")
            tmoy = env_cols[0].number_input("Temp. moyenne (°C)", -15.0, 45.0, 15.0, format="%.1f")
            pm25 = env_cols[1].number_input("PM2.5 (µg/m³)", 0.0, 100.0, 15.0, format="%.1f")
            pm10 = env_cols[0].number_input("PM10 (µg/m³)", 0.0, 150.0, 25.0, format="%.1f")
            no2 = env_cols[1].number_input("NO2 (µg/m³)", 0.0, 200.0, 30.0, format="%.1f")
            o3 = env_cols[0].number_input("O3 (µg/m³)", 0.0, 200.0, 50.0, format="%.1f")

            st.markdown("#### 💉 Vaccination")
            vaccination = st.number_input("Doses administrées (total)", 0, 80000000, 5000000)

            submitted = st.form_submit_button("Lancer la prédiction")

            if submitted:
                input_data = {
                    'retail_and_recreation': retail, 'grocery_and_pharmacy': grocery, 'parks': parks,
                    'transit_stations': transit, 'workplaces': workplaces, 'residential': residential,
                    'TMin (°C)': tmin, 'TMax (°C)': tmax, 'TMoy (°C)': tmoy,
                    'pm25': pm25, 'pm10': pm10, 'o3': o3, 'no2': no2, 'Vaccination': vaccination
                }
                
                try:
                    input_df = pd.DataFrame([input_data], columns=FEATURES)
                    input_scaled = scaler.transform(input_df)
                    pred = model.predict(input_scaled)[0]
                    pred = max(0, pred) # Assurer une prédiction non-négative

                    st.success("### 🎯 Résultats de la simulation")
                    
                    res_cols = st.columns(2)
                    res_cols[0].metric("Cas hebdomadaires prédits", f"{int(pred):,}")
                    
                    alert_status = "🔴 Alerte" if pred > seuil_alerte else "🟡 Modéré" if pred > seuil_alerte*0.7 else "🟢 Stable"
                    res_cols[1].metric("Niveau d'alerte", alert_status)
                    
                    prog_value = min(pred / seuil_alerte if seuil_alerte > 0 else 1.0, 1.0)
                    st.progress(prog_value, text=f"{prog_value*100:.1f}% du seuil d'alerte ({int(seuil_alerte):,} cas)")
                    
                    if pred > seuil_alerte:
                        st.warning("#### Recommandations pour un niveau d'alerte élevé :\n- Envisager le renforcement des mesures sanitaires (masques, distanciation).\n- Lancer des campagnes de communication pour limiter les rassemblements.\n- Accélérer la vaccination de rappel.")
                    elif pred > seuil_alerte * 0.5:
                        st.info("#### Recommandations pour un niveau modéré :\n- Renforcer la surveillance épidémiologique.\n- Promouvoir activement la vaccination et les gestes barrières.\n- Sensibiliser les populations sur les risques.")
                    else:
                        st.success("#### Situation sous contrôle :\n- Maintenir la surveillance et la communication préventive.\n- Continuer les efforts de vaccination.")

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction de la simulation : {str(e)}")