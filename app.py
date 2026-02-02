import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import numpy as np

st.set_page_config(page_title="Carte e-commerce", layout="wide")
st.title("📍 Boutiques & commandes e-commerce")

# =========================
# SESSION STATE (appliquer filtres)
# =========================
if "applied_annee" not in st.session_state:
    st.session_state.applied_annee = None
if "applied_code_magasin" not in st.session_state:
    st.session_state.applied_code_magasin = "magBouq"

# =========================
# DB CONNECT
# =========================
conn = psycopg2.connect(
    host=st.secrets["DB_HOST"],
    port=st.secrets["DB_PORT"],
    dbname=st.secrets["DB_NAME"],
    user=st.secrets["DB_USER"],
    password=st.secrets["DB_PASSWORD"],
    sslmode="require",
)

# =========================
# LOAD DATA
# =========================
df_magasin = pd.read_sql("""
    SELECT code_magasin, nom_magasin, latitude, longitude
    FROM public.ref_magasin
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
""", conn)

df_magasin_ecom = pd.read_sql("""
    SELECT code_magasin
    FROM public.ref_magasin_ecommerce
""", conn)

df_cmd_all = pd.read_sql("""
    SELECT
      code_commande,
      total_commande,
      code_magasin,
      latitude,
      longitude,
      LEFT(code_commande, 4) AS annee
    FROM public.commande_ecommerce
    WHERE latitude IS NOT NULL
      AND longitude IS NOT NULL
""", conn)

# =========================
# FILTRES UI + BOUTON APPLIQUER
# =========================
st.subheader("🎛️ Filtres")

annees = sorted(df_cmd_all["annee"].dropna().unique().tolist())
if not annees:
    st.error("Aucune année détectée (LEFT(code_commande, 4)).")
    conn.close()
    st.stop()

# Année par défaut = 2025 si dispo, sinon dernière année
annee_defaut = "2025" if "2025" in annees else annees[-1]

# Si première fois, on initialise avec défaut
if st.session_state.applied_annee is None:
    st.session_state.applied_annee = annee_defaut


# Filtre "en attente" (pas appliqué tant que bouton pas cliqué)
pending_annee = st.selectbox(
    "📅 Année",
    annees,
    index=annees.index(st.session_state.applied_annee) if st.session_state.applied_annee in annees else 0,
    key="pending_annee"
)

# Codes magasin dispo selon l'année "pending"
df_cmd_pending = df_cmd_all[df_cmd_all["annee"] == pending_annee]
codes_magasin = sorted(df_cmd_pending["code_magasin"].dropna().unique().tolist())

if not codes_magasin:
    st.warning(f"Aucun code magasin disponible pour l'année {pending_annee}.")
    conn.close()
    st.stop()

default_index = codes_magasin.index(st.session_state.applied_code_magasin) if st.session_state.applied_code_magasin in codes_magasin else (
    codes_magasin.index("magBouq") if "magBouq" in codes_magasin else 0
)

pending_code_magasin = st.selectbox(
    "🏬 Code magasin",
    codes_magasin,
    index=default_index,
    key="pending_code_magasin"
)

# Bouton Appliquer
apply_clicked = st.button("✅ Appliquer les filtres")

if apply_clicked:
    st.session_state.applied_annee = pending_annee
    st.session_state.applied_code_magasin = pending_code_magasin

# Filtres effectivement appliqués
selected_annee = st.session_state.applied_annee
selected_code_magasin = st.session_state.applied_code_magasin

# =========================
# FILTRE FINAL COMMANDES (APPLIQUÉ)
# =========================
df_cmd = df_cmd_all[
    (df_cmd_all["annee"] == selected_annee) &
    (df_cmd_all["code_magasin"] == selected_code_magasin)
].copy()

# =========================
# PREP
# =========================
df_magasin["code_magasin"] = df_magasin["code_magasin"].astype(str)
df_magasin_ecom["code_magasin"] = df_magasin_ecom["code_magasin"].astype(str)

ecom_set = set(df_magasin_ecom["code_magasin"].dropna().tolist())

df_magasin["categorie"] = df_magasin["code_magasin"].apply(
    lambda x: "Magasin e-commerce" if x in ecom_set else "Magasin"
)

df_magasin["hover"] = df_magasin.apply(
    lambda r: f"{(r['nom_magasin'] or '').strip()} ({r['code_magasin']})<br>{r['categorie']}",
    axis=1,
)

df_cmd["hover"] = df_cmd.apply(
    lambda r: f"Commande: {r['code_commande']}<br>"
              f"Année: {r['annee']}<br>"
              f"Code magasin: {r['code_magasin']}<br>"
              f"Total: {r['total_commande']}",
    axis=1,
)

# =========================
# CENTER MAP
# =========================
all_lat = pd.concat([df_magasin["latitude"], df_cmd["latitude"]], ignore_index=True)
all_lon = pd.concat([df_magasin["longitude"], df_cmd["longitude"]], ignore_index=True)

center_lat = float(all_lat.mean()) if len(all_lat) else 46.6
center_lon = float(all_lon.mean()) if len(all_lon) else 2.2

# =========================
# MAP
# =========================
fig = go.Figure()

# COMMANDES (dessous)
if not df_cmd.empty:
    fig.add_trace(go.Scattermapbox(
        name=f"Commandes {selected_annee} ({selected_code_magasin})",
        lat=df_cmd["latitude"],
        lon=df_cmd["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(size=8, color="red", opacity=0.75),
        text=df_cmd["hover"],
        hoverinfo="text",
    ))
else:
    st.info(f"Aucune commande pour {selected_code_magasin} en {selected_annee}")

# MAGASINS (dessus)
MAG_SIZE = 12
df_mag_blue = df_magasin[df_magasin["categorie"] == "Magasin"]
df_mag_yellow = df_magasin[df_magasin["categorie"] == "Magasin e-commerce"]

if not df_mag_blue.empty:
    fig.add_trace(go.Scattermapbox(
        name="Magasins",
        lat=df_mag_blue["latitude"],
        lon=df_mag_blue["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(size=MAG_SIZE, color="blue", opacity=0.95),
        text=df_mag_blue["hover"],
        hoverinfo="text",
    ))

if not df_mag_yellow.empty:
    fig.add_trace(go.Scattermapbox(
        name="Magasins e-commerce",
        lat=df_mag_yellow["latitude"],
        lon=df_mag_yellow["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(size=MAG_SIZE, color="yellow", opacity=0.95),
        text=df_mag_yellow["hover"],
        hoverinfo="text",
    ))

fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=5,
    ),
    height=750,
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# CLOSE DB
# =========================
conn.close()
