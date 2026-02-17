import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import numpy as np
import math

from auth import require_auth, logout

st.set_page_config(page_title="Carte e-commerce", layout="wide")

# =========================
# AUTH
# =========================
require_auth()

with st.sidebar:
    st.write("✅ Connecté")
    user_email = (st.session_state.get("sb_user") or {}).get("email", "")
    if user_email:
        st.caption(user_email)
    if st.button("Se déconnecter"):
        logout()

st.title("📍 Boutiques & commandes e-commerce")

# =========================
# HELPERS (sans sklearn)
# =========================
EARTH_R_KM = 6371.0088

def circle_latlon(center_lat: float, center_lon: float, radius_km: float, n_points: int = 48):
    lat1 = math.radians(center_lat)
    lon1 = math.radians(center_lon)
    d = radius_km / EARTH_R_KM

    lats, lons = [], []
    for i in range(n_points + 1):
        brng = 2 * math.pi * (i / n_points)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng)
        )
        lon2 = lon1 + math.atan2(
            math.sin(brng) * math.sin(d) * math.cos(lat1),
            math.cos(d) - math.sin(lat1) * math.sin(lat2),
        )
        lats.append(math.degrees(lat2))
        lons.append(math.degrees(lon2))
    return lats, lons

def norm_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def norm_cp(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.replace({"nan": None, "None": None, "": None})
    return out

# =========================
# SESSION STATE
# =========================
if "applied_annee" not in st.session_state:
    st.session_state.applied_annee = None
if "applied_code_magasin" not in st.session_state:
    st.session_state.applied_code_magasin = "MAGBOUQ"
if "applied_radius_km" not in st.session_state:
    st.session_state.applied_radius_km = 10
if "applied_circle_codes" not in st.session_state:
    st.session_state.applied_circle_codes = []
if "applied_departements" not in st.session_state:
    st.session_state.applied_departements = []
# ✅ NEW: afficher / cacher les rayons e-commerce
if "applied_show_ecom_radii" not in st.session_state:
    st.session_state.applied_show_ecom_radii = True

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
    SELECT code_magasin, nom_magasin, latitude, longitude, cp
    FROM public.ref_magasin
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
""", conn)

# ✅ On récupère aussi le rayon e-commerce (en mètres)
df_magasin_ecom = pd.read_sql("""
    SELECT
      code_magasin,
      COALESCE(rayon_en_metres, 0) AS rayon_en_metres
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
# PREP TYPES + NORMALISATION
# =========================
df_magasin["code_magasin"] = norm_code(df_magasin["code_magasin"])
df_magasin["nom_magasin"] = df_magasin["nom_magasin"].astype(str)
df_magasin["cp"] = norm_cp(df_magasin["cp"])
df_magasin["departement"] = df_magasin["cp"].apply(
    lambda x: str(x)[:2] if isinstance(x, str) and len(x) >= 2 else None
)
df_magasin["latitude"] = pd.to_numeric(df_magasin["latitude"], errors="coerce")
df_magasin["longitude"] = pd.to_numeric(df_magasin["longitude"], errors="coerce")
df_magasin = df_magasin.dropna(subset=["latitude", "longitude"]).copy()

df_magasin_ecom["code_magasin"] = norm_code(df_magasin_ecom["code_magasin"])
df_magasin_ecom["rayon_en_metres"] = pd.to_numeric(df_magasin_ecom["rayon_en_metres"], errors="coerce").fillna(0.0)
df_magasin_ecom["rayon_km"] = (df_magasin_ecom["rayon_en_metres"] / 1000.0).round(3)

# ✅ set des magasins e-commerce
ecom_set = set(df_magasin_ecom["code_magasin"].dropna().tolist())

# ✅ on injecte le rayon e-commerce dans df_magasin
df_magasin = df_magasin.merge(
    df_magasin_ecom[["code_magasin", "rayon_km"]],
    on="code_magasin",
    how="left"
)
df_magasin["rayon_km"] = df_magasin["rayon_km"].fillna(0.0)

df_cmd_all["annee"] = df_cmd_all["annee"].astype(str)
df_cmd_all["code_magasin"] = norm_code(df_cmd_all["code_magasin"])
df_cmd_all["latitude"] = pd.to_numeric(df_cmd_all["latitude"], errors="coerce")
df_cmd_all["longitude"] = pd.to_numeric(df_cmd_all["longitude"], errors="coerce")
df_cmd_all["total_commande"] = pd.to_numeric(df_cmd_all["total_commande"], errors="coerce").fillna(0.0)
df_cmd_all = df_cmd_all.dropna(subset=["latitude", "longitude"]).copy()

df_magasin["categorie"] = df_magasin["code_magasin"].apply(
    lambda x: "Magasin e-commerce" if x in ecom_set else "Magasin"
)

df_magasin["hover"] = df_magasin.apply(
    lambda r: f"{(r['nom_magasin'] or '').strip()} ({r['code_magasin']})<br>{r['categorie']}",
    axis=1,
)

# =========================
# FILTRES UI + BOUTON APPLIQUER
# =========================
st.subheader("🎛️ Filtres")

annees = sorted(df_cmd_all["annee"].dropna().unique().tolist())
if not annees:
    st.error("Aucune année détectée (LEFT(code_commande, 4)).")
    conn.close()
    st.stop()

annee_defaut = "2025" if "2025" in annees else annees[-1]
if st.session_state.applied_annee is None:
    st.session_state.applied_annee = annee_defaut

pending_annee = st.selectbox(
    "📅 Année (sert au calcul du potentiel)",
    annees,
    index=annees.index(st.session_state.applied_annee) if st.session_state.applied_annee in annees else 0,
    key="pending_annee",
)

df_cmd_pending = df_cmd_all[df_cmd_all["annee"] == pending_annee]
codes_magasin = sorted(df_cmd_pending["code_magasin"].dropna().unique().tolist())
if not codes_magasin:
    st.warning(f"Aucun code magasin disponible pour l'année {pending_annee}.")
    conn.close()
    st.stop()

df_mag_label = df_magasin[["code_magasin", "nom_magasin"]].drop_duplicates().copy()
df_mag_label = df_mag_label[df_mag_label["code_magasin"].isin(codes_magasin)].copy()
df_mag_label["label"] = df_mag_label.apply(lambda r: f"{r['code_magasin']} - {r['nom_magasin']}", axis=1)

labels_red = sorted(df_mag_label["label"].tolist())
label_to_code_red = dict(zip(df_mag_label["label"], df_mag_label["code_magasin"]))

default_label_red = None
for lbl, code in label_to_code_red.items():
    if code == st.session_state.applied_code_magasin:
        default_label_red = lbl
        break

pending_label_magasin = st.selectbox(
    "🏬 Magasin (pour afficher ses commandes en rouge)",
    labels_red,
    index=labels_red.index(default_label_red) if default_label_red in labels_red else 0,
    key="pending_code_magasin_label",
)
pending_code_magasin = label_to_code_red[pending_label_magasin]

# ✅✅✅ DERNIÈRE MODIF DEMANDÉE : toggle placé juste sous le select magasin rouge
pending_show_ecom_radii = st.checkbox(
    "🟢 Afficher les rayons des magasins e-commerce",
    value=bool(st.session_state.applied_show_ecom_radii),
    key="pending_show_ecom_radii",
)

pending_radius_km = st.slider(
    "🔵 Rayon (km) appliqué aux magasins sélectionnés (cercles bleus)",
    min_value=1,
    max_value=50,
    value=int(st.session_state.applied_radius_km),
    step=1,
    key="pending_radius_km",
)

deps = sorted([d for d in df_magasin["departement"].dropna().unique().tolist() if len(str(d)) == 2])
pending_departements = st.multiselect(
    "🗺️ Départements (2 premiers chiffres du CP) — filtre la liste des magasins",
    options=deps,
    default=st.session_state.applied_departements,
    key="pending_departements",
)

# ✅ IMPORTANT : on EXCLUT les ECOM de la liste des cercles manuels
df_all_options = df_magasin[["code_magasin", "nom_magasin", "categorie", "departement"]].drop_duplicates().copy()
df_all_options = df_all_options[df_all_options["categorie"] != "Magasin e-commerce"].copy()

if pending_departements:
    df_all_options = df_all_options[df_all_options["departement"].isin(pending_departements)].copy()

df_all_options["label"] = df_all_options.apply(
    lambda r: f"{r['code_magasin']} - {r['nom_magasin']} (NON-ECOM)",
    axis=1
)

labels_sorted = sorted(df_all_options["label"].unique().tolist())
label_to_code_circle = dict(zip(df_all_options["label"], df_all_options["code_magasin"]))

current_codes = st.session_state.applied_circle_codes
default_labels = []
if current_codes:
    code_to_label_circle = {v: k for k, v in label_to_code_circle.items()}
    default_labels = [code_to_label_circle[c] for c in current_codes if c in code_to_label_circle]

pending_circle_labels = st.multiselect(
    "⭕ Magasins NON-ECOM pour afficher le cercle (max 40)",
    options=labels_sorted,
    default=default_labels,
    max_selections=40,
    key="pending_circle_labels",
)

apply_clicked = st.button("✅ Appliquer les filtres")
if apply_clicked:
    st.session_state.applied_annee = pending_annee
    st.session_state.applied_show_ecom_radii = bool(pending_show_ecom_radii)
    st.session_state.applied_code_magasin = pending_code_magasin
    st.session_state.applied_radius_km = pending_radius_km
    st.session_state.applied_departements = pending_departements
    st.session_state.applied_circle_codes = [label_to_code_circle[l] for l in pending_circle_labels]

selected_annee = st.session_state.applied_annee
selected_show_ecom_radii = bool(st.session_state.applied_show_ecom_radii)
selected_code_magasin = st.session_state.applied_code_magasin
selected_radius_km = float(st.session_state.applied_radius_km)
selected_circle_codes = st.session_state.applied_circle_codes

# =========================
# COMMANDES ROUGES (affichage)
# =========================
df_cmd_red = df_cmd_all[
    (df_cmd_all["annee"] == selected_annee) &
    (df_cmd_all["code_magasin"] == selected_code_magasin)
].copy()

df_cmd_red["hover"] = df_cmd_red.apply(
    lambda r: f"Commande: {r['code_commande']}<br>"
              f"Année: {r['annee']}<br>"
              f"Code magasin: {r['code_magasin']}<br>"
              f"Total: {float(r['total_commande']):,.2f} €".replace(",", " "),
    axis=1,
)

# =========================
# POTENTIEL (commandes dans le cercle)
# =========================
df_orders_year = df_cmd_all[df_cmd_all["annee"] == selected_annee].copy()

df_orders_ecom = df_orders_year[df_orders_year["code_magasin"].isin(ecom_set)].copy()
orders_used = df_orders_ecom if not df_orders_ecom.empty else df_orders_year

df_all_mag = df_magasin.copy()
df_all_mag["potential_nb_cmd"] = 0
df_all_mag["potential_total_cmd"] = 0.0

if not orders_used.empty and not df_all_mag.empty:
    o_lat = orders_used["latitude"].to_numpy(dtype=float)
    o_lon = orders_used["longitude"].to_numpy(dtype=float)
    o_total = orders_used["total_commande"].to_numpy(dtype=float)

    batch_size = 80
    nb_centers = len(df_all_mag)
    pot_counts = np.zeros(nb_centers, dtype=np.int32)
    pot_sums = np.zeros(nb_centers, dtype=np.float64)

    for start in range(0, nb_centers, batch_size):
        end = min(start + batch_size, nb_centers)
        b = df_all_mag.iloc[start:end]

        b_lat = b["latitude"].to_numpy(dtype=float)[:, None]
        b_lon = b["longitude"].to_numpy(dtype=float)[:, None]

        lat1 = np.radians(b_lat)
        lon1 = np.radians(b_lon)
        lat2 = np.radians(o_lat[None, :])
        lon2 = np.radians(o_lon[None, :])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        dist = 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(a))

        in_circle = dist <= selected_radius_km
        pot_counts[start:end] = in_circle.sum(axis=1).astype(np.int32)
        pot_sums[start:end] = (in_circle * o_total[None, :]).sum(axis=1).astype(np.float64)

    df_all_mag["potential_nb_cmd"] = pot_counts
    df_all_mag["potential_total_cmd"] = pot_sums

df_all_mag["hover_potential"] = df_all_mag.apply(
    lambda r: (
        f"<b>{(r['nom_magasin'] or '').strip()}</b><br>"
        f"Code: {r['code_magasin']}<br>"
        f"Type: {'ECOM' if r['categorie']=='Magasin e-commerce' else 'NON-ECOM'}<br>"
        f"Département: {r['departement'] if pd.notna(r['departement']) else '-'}<br>"
        f"Rayon bleu: {int(selected_radius_km)} km (année {selected_annee})<br>"
        f"<br><b>Potentiel (commandes dans le cercle)</b><br>"
        f"Nb commandes: {int(r['potential_nb_cmd'])}<br>"
        f"Montant total: {float(r['potential_total_cmd']):,.2f} €"
    ).replace(",", " "),
    axis=1,
)

df_selected = df_all_mag[df_all_mag["code_magasin"].isin(selected_circle_codes)].copy()

# =========================
# CENTER MAP
# =========================
all_lat = pd.concat([df_magasin["latitude"], df_orders_year["latitude"]], ignore_index=True)
all_lon = pd.concat([df_magasin["longitude"], df_orders_year["longitude"]], ignore_index=True)
center_lat = float(all_lat.mean()) if len(all_lat) else 46.6
center_lon = float(all_lon.mean()) if len(all_lon) else 2.2

# =========================
# MAP
# =========================
fig = go.Figure()

if not df_cmd_red.empty:
    fig.add_trace(go.Scattermapbox(
        name=f"Commandes {selected_annee} ({selected_code_magasin})",
        lat=df_cmd_red["latitude"],
        lon=df_cmd_red["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(size=8, color="red", opacity=0.75),
        text=df_cmd_red["hover"],
        hovertemplate="%{text}<extra></extra>",
    ))

df_mag_yellow = df_magasin[df_magasin["categorie"] == "Magasin e-commerce"]
if not df_mag_yellow.empty:
    fig.add_trace(go.Scattermapbox(
        name="Magasins e-commerce",
        lat=df_mag_yellow["latitude"],
        lon=df_mag_yellow["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(size=12, color="yellow", opacity=0.95),
        text=df_mag_yellow["hover"],
        hovertemplate="%{text}<extra></extra>",
    ))

df_non_ecom = df_magasin[df_magasin["categorie"] == "Magasin"]
if not df_non_ecom.empty:
    df_non_ecom_hover = df_all_mag[df_all_mag["categorie"] == "Magasin"][["code_magasin", "hover_potential"]].copy()
    df_non_ecom_plot = df_non_ecom.merge(df_non_ecom_hover, on="code_magasin", how="left")
    fig.add_trace(go.Scattermapbox(
        name="Magasins NON e-commerce",
        lat=df_non_ecom_plot["latitude"],
        lon=df_non_ecom_plot["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(size=12, color="orange", opacity=0.95),
        text=df_non_ecom_plot["hover_potential"].fillna(df_non_ecom_plot["hover"]),
        hovertemplate="%{text}<extra></extra>",
    ))

# =========================
# ✅ CERCLES E-COM (VERT) — AFFICHAGE ON/OFF
# =========================
if selected_show_ecom_radii:
    df_ecom_centers = df_all_mag[
        (df_all_mag["categorie"] == "Magasin e-commerce") &
        (df_all_mag["rayon_km"] > 0)
    ].copy()

    for r in df_ecom_centers.itertuples(index=False):
        radius_km = float(r.rayon_km)
        clats, clons = circle_latlon(float(r.latitude), float(r.longitude), radius_km, n_points=48)

        hover_circle = (
            f"<b>Zone e-commerce</b><br>"
            f"{(r.nom_magasin or '').strip()} ({r.code_magasin})<br>"
            f"Rayon: {int(radius_km * 1000)} m ({radius_km:.2f} km)<br>"
        ).replace(",", " ")

        fig.add_trace(go.Scattermapbox(
            showlegend=False,
            lat=clats,
            lon=clons,
            mode="lines",
            fill="toself",
            fillcolor="rgba(0, 255, 128, 0.14)",
            line=dict(width=1.5, color="rgba(0, 200, 100, 0.95)"),
            text=[hover_circle] * len(clats),
            hovertemplate="%{text}<extra></extra>",
        ))

# =========================
# CERCLES BLEUS — magasins NON-ECOM sélectionnés
# =========================
if not df_selected.empty and selected_radius_km > 0:
    for r in df_selected.itertuples(index=False):
        clats, clons = circle_latlon(float(r.latitude), float(r.longitude), selected_radius_km, n_points=48)

        hover_circle = (
            f"<b>Zone</b><br>"
            f"{(r.nom_magasin or '').strip()} ({r.code_magasin})<br>"
            f"Type: {'ECOM' if r.categorie=='Magasin e-commerce' else 'NON-ECOM'}<br>"
            f"Département: {r.departement if pd.notna(r.departement) else '-'}<br>"
            f"Rayon bleu: {int(selected_radius_km)} km (année {selected_annee})<br>"
            f"<br><b>Potentiel</b><br>"
            f"Nb commandes: {int(r.potential_nb_cmd)}<br>"
            f"Montant total: {float(r.potential_total_cmd):,.2f} €"
        ).replace(",", " ")

        fig.add_trace(go.Scattermapbox(
            showlegend=False,
            lat=clats,
            lon=clons,
            mode="lines",
            fill="toself",
            fillcolor="rgba(0, 102, 255, 0.12)",
            line=dict(width=1.5, color="rgba(0, 102, 255, 0.9)"),
            text=[hover_circle] * len(clats),
            hovertemplate="%{text}<extra></extra>",
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
# TABLEAU EN BAS (seulement cercles bleus sélectionnés)
# =========================
st.subheader("📊 Potentiel des magasins sélectionnés")

if df_selected.empty:
    st.info("Aucun magasin sélectionné pour afficher le tableau.")
else:
    df_table_num = df_selected[[
        "code_magasin",
        "nom_magasin",
        "categorie",
        "departement",
        "potential_nb_cmd",
        "potential_total_cmd",
    ]].copy()

    df_table_num["panier_moyen_potentiel"] = df_table_num.apply(
        lambda r: float(r["potential_total_cmd"]) / int(r["potential_nb_cmd"]) if int(r["potential_nb_cmd"]) > 0 else 0.0,
        axis=1
    )

    df_table_num.sort_values(
        by=["potential_total_cmd", "potential_nb_cmd"],
        ascending=False,
        inplace=True
    )

    df_table = df_table_num.rename(columns={
        "code_magasin": "Code magasin",
        "nom_magasin": "Nom magasin",
        "categorie": "Type",
        "departement": "Département",
        "potential_nb_cmd": "Nb commandes potentielles",
        "potential_total_cmd": "CA potentiel",
        "panier_moyen_potentiel": "Panier moyen potentiel",
    }).copy()

    df_table["CA potentiel"] = df_table["CA potentiel"].round(2).map(
        lambda x: f"{x:,.2f} €".replace(",", " ")
    )
    df_table["Panier moyen potentiel"] = df_table["Panier moyen potentiel"].round(2).map(
        lambda x: f"{x:,.2f} €".replace(",", " ")
    )

    total_pot = float(df_table_num["potential_total_cmd"].sum())
    total_nb = int(df_table_num["potential_nb_cmd"].sum())
    total_pm = (total_pot / total_nb) if total_nb > 0 else 0.0

    st.caption(
        f"Rayon bleu: {int(selected_radius_km)} km — Année: {selected_annee} — "
        f"Magasins: {len(df_table_num)} — "
        f"CA potentiel total: {total_pot:,.2f} € — "
        f"Nb commandes: {total_nb} — "
        f"Panier moyen: {total_pm:,.2f} €"
        .replace(",", " ")
    )

    st.dataframe(df_table, use_container_width=True, hide_index=True)

conn.close()
