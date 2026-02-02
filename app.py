import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import numpy as np  # (pas utilisé ici, mais je le laisse comme dans ton code)

st.set_page_config(page_title="Carte e-commerce", layout="wide")
st.title("📍 Boutiques & commandes e-commerce — Année 2025")

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

# ⚠️ Commandes filtrées sur 2025 (4 premiers caractères du code_commande)
df_cmd_all = pd.read_sql("""
    SELECT code_commande, total_commande, code_magasin, latitude, longitude
    FROM public.commande_ecommerce
    WHERE latitude IS NOT NULL
      AND longitude IS NOT NULL
      AND LEFT(code_commande, 4) = '2025'
""", conn)

# =========================
# FILTRE CODE_MAGASIN (COMMANDES) - défaut: magBouq
# =========================
codes_magasin = sorted(df_cmd_all["code_magasin"].dropna().unique().tolist())

if not codes_magasin:
    st.warning("Aucune commande 2025 trouvée (LEFT(code_commande,4)='2025').")
    conn.close()
    st.stop()

default_index = codes_magasin.index("magBouq") if "magBouq" in codes_magasin else 0

selected_code_magasin = st.selectbox(
    "🏬 Filtrer les commandes (2025) par code magasin",
    codes_magasin,
    index=default_index
)

df_cmd = df_cmd_all[df_cmd_all["code_magasin"] == selected_code_magasin].copy()

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

# COMMANDES (dessous, filtrées, 2025)
if not df_cmd.empty:
    fig.add_trace(go.Scattermapbox(
        name=f"Commandes 2025 ({selected_code_magasin})",
        lat=df_cmd["latitude"],
        lon=df_cmd["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(size=8, color="red", opacity=0.75),
        text=df_cmd["hover"],
        hoverinfo="text",
    ))
else:
    st.info(f"Aucune commande 2025 pour le code magasin : {selected_code_magasin}")

# MAGASINS (dessus)
MAG_SIZE = 12
df_mag_blue = df_magasin[df_magasin["categorie"] == "Magasin"]
df_mag_green = df_magasin[df_magasin["categorie"] == "Magasin e-commerce"]

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

if not df_mag_green.empty:
    fig.add_trace(go.Scattermapbox(
        name="Magasins e-commerce",
        lat=df_mag_green["latitude"],
        lon=df_mag_green["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=MAG_SIZE,
            color="yellow",   # ⬅️ CHANGEMENT ICI
            opacity=0.95
        ),
        text=df_mag_green["hover"],
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
