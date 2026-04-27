import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                              accuracy_score, classification_report)
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Premier League ML Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1a;
    color: #e2e8f0;
}
.main { background-color: #0a0f1a; }
.block-container { padding: 2rem 3rem; }

.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: clamp(1.8rem, 5vw, 4rem);
    letter-spacing: 3px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
    line-height: 1.1;
    white-space: nowrap;
    overflow: visible;
    width: 100%;
}
.hero-sub {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}
.section-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 2.2rem;
    letter-spacing: 3px;
    color: #38bdf8;
    border-left: 4px solid #818cf8;
    padding-left: 1rem;
    margin: 2rem 0 1rem 0;
}
.kpi-card {
    background: linear-gradient(135deg, #37003c 0%, #1a0020 100%);
    border: 1px solid #00ff85;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 15px rgba(0,255,133,0.1);
}
.kpi-value {
    font-family: 'Bebas Neue', cursive;
    font-size: 2.8rem;
    letter-spacing: 2px;
    color: #00ff85;
    line-height: 1;
}
.kpi-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin-top: 0.3rem;
}
.insight-box {
    background: #0f172a;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #94a3b8;
}
.benchmark-win {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #16a34a;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.benchmark-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4ade80;
}
.benchmark-value {
    font-family: 'Bebas Neue', cursive;
    font-size: 3rem;
    color: #4ade80;
}
.stTabs [data-baseweb="tab-list"] {
    background: #37003c;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: white !important;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700 !important;
    letter-spacing: 1px;
    font-size: 0.8rem;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #00ff85 !important;
    color: #37003c !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ── PANTALLA DE BIENVENIDA ──────────────────────────────
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

if not st.session_state.show_dashboard:
    with open("welcome.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=False)
    st.markdown("<br>", unsafe_allow_html=True)
    col = st.columns([1, 2, 1])[1]
    with col:
        if st.button("ENTRAR AL DASHBOARD", use_container_width=True, type="primary"):
            st.session_state.show_dashboard = True
            st.rerun()
    st.stop()
# ────────────────────────────────────────────────────────

# ─────────────────────────────────────────────
# CARGA DE DATOS REALES
# ─────────────────────────────────────────────
BASE = "https://premier.72-60-245-2.sslip.io"

@st.cache_data(ttl=3600, show_spinner="Cargando datos de la Premier League...")
def load_data():
    matches = pd.read_csv(f"{BASE}/export/matches")
    players = pd.read_csv(f"{BASE}/export/players")
    shots_raw = requests.get(f"{BASE}/events?is_shot=true&limit=10000").json()
    shots = pd.DataFrame(shots_raw['events'])

    # Fechas
    matches['date'] = pd.to_datetime(matches['date'], format='%d/%m/%Y', errors='coerce')
    matches['total_goals'] = matches['fthg'] + matches['ftag']

    # Probabilidades implícitas
    for prefix in ['b365', 'bw', 'max', 'avg']:
        cols = [f'{prefix}h', f'{prefix}d', f'{prefix}a']
        if all(c in matches.columns for c in cols):
            inv = matches[cols].apply(lambda x: 1/x)
            total = inv.sum(axis=1)
            matches[f'implied_prob_h_{prefix}'] = inv[cols[0]] / total
            matches[f'implied_prob_d_{prefix}'] = inv[cols[1]] / total
            matches[f'implied_prob_a_{prefix}'] = inv[cols[2]] / total

    matches['implied_prob_h'] = matches['implied_prob_h_b365']
    matches['implied_prob_d'] = matches['implied_prob_d_b365']
    matches['implied_prob_a'] = matches['implied_prob_a_b365']

    return matches, players, shots

matches, players, shots = load_data()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — SHOTS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Calculando features...")
def build_shots_features(shots_df):
    df = shots_df.copy()
    df['distancia'] = np.sqrt((100 - df['x'])**2 + (50 - df['y'])**2)

    def _angulo(row):
        a = np.sqrt((100 - row['x'])**2 + (45.2 - row['y'])**2)
        b = np.sqrt((100 - row['x'])**2 + (54.8 - row['y'])**2)
        c = 9.6
        return np.arccos(np.clip((a**2 + b**2 - c**2) / (2*a*b), -1, 1))

    df['angulo_rad'] = df.apply(_angulo, axis=1)
    df['angulo_grados'] = np.degrees(df['angulo_rad'])

    angulo_grados = np.degrees(df['angulo_rad'])
    df['angulo_zona'] = pd.cut(
        angulo_grados,
        bins=[0, 11, 28, 57, 86, 180],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    q = df['qualifiers'].astype(str)
    df['is_big_chance'] = q.str.contains('BigChance',  na=False, regex=False).astype(int)
    df['is_header']     = q.str.contains('Head',       na=False, regex=False).astype(int)
    df['is_penalty']    = q.str.contains('Penalty',    na=False, regex=False).astype(int)
    df['is_fast_break'] = q.str.contains('FastBreak',  na=False, regex=False).astype(int)
    df['first_touch']   = q.str.contains('FirstTouch', na=False, regex=False).astype(int)
    df['en_area_grande'] = (df['x'] >= 83).astype(int)
    df['en_area_chica']  = (df['x'] >= 94).astype(int)
    df['is_final_minutes'] = (df['minute'] > 85).astype(int)

    if 'goal_mouth_z' in df.columns:
        df['goal_mouth_z'] = pd.to_numeric(df['goal_mouth_z'], errors='coerce')
        df['goal_mouth_z'] = df['goal_mouth_z'].fillna(df['goal_mouth_z'].median())

    return df

shots_final = build_shots_features(shots)

# ─────────────────────────────────────────────
# ENTRENAR MODELOS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Entrenando modelos...")
def train_all_models(shots_df, matches_df):
    # — MODELO xG —
    features_xg = ['angulo_zona', 'is_big_chance', 'is_penalty', 'is_header',
                    'is_fast_break', 'first_touch', 'en_area_grande',
                    'en_area_chica', 'is_final_minutes']
    if 'goal_mouth_z' in shots_df.columns:
        features_xg.append('goal_mouth_z')

    X_xg = shots_df[features_xg].fillna(0)
    y_xg = shots_df['is_goal'].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_xg, y_xg, test_size=0.2, random_state=42, stratify=y_xg)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    model_xg = LogisticRegression(penalty='l2', C=1.0, max_iter=3000, random_state=42)
    model_xg.fit(X_tr_sc, y_tr)

    y_pred_xg = model_xg.predict(X_te_sc)
    y_prob_xg = model_xg.predict_proba(X_te_sc)[:, 1]

    # RF para xG
    ratio = (y_xg==0).sum()/(y_xg==1).sum()
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_xg.fillna(0), y_xg, test_size=0.2, random_state=42, stratify=y_xg)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr_r, y_tr_r)
    rf_probs = rf.predict_proba(X_te_r)[:, 1]

    # — MODELO MATCH PREDICTOR —
    feat_match = ['b365h', 'b365d', 'b365a',
                  'implied_prob_h', 'implied_prob_d', 'implied_prob_a']
    X_log = matches_df[feat_match].dropna()
    y_log = matches_df.loc[X_log.index, 'ftr']

    log_match = LogisticRegression(solver='lbfgs', max_iter=2000)
    acc_cv = cross_val_score(log_match, X_log, y_log, cv=5)
    y_pred_match = cross_val_predict(log_match, X_log, y_log, cv=5)
    log_match.fit(X_log, y_log)

    # — RIDGE GOLES —
    matches_sorted = matches_df.sort_values('date').copy()
    for team_col, goals_col, new_col in [
        ('home_team', 'fthg', 'home_avg_scored'),
        ('home_team', 'ftag', 'home_avg_conceded'),
        ('away_team', 'ftag', 'away_avg_scored'),
        ('away_team', 'fthg', 'away_avg_conceded'),
    ]:
        matches_sorted[new_col] = (
            matches_sorted.groupby(team_col)[goals_col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        )
    matches_sorted['ataque_vs_defensa_h'] = (matches_sorted['home_avg_scored'] +
                                              matches_sorted['away_avg_conceded'])
    matches_sorted['ataque_vs_defensa_a'] = (matches_sorted['away_avg_scored'] +
                                              matches_sorted['home_avg_conceded'])
    matches_sorted['over_proxy'] = 1/matches_sorted['avgh'] + 1/matches_sorted['avga']

    feat_goles = ['home_avg_scored','home_avg_conceded','away_avg_scored',
                  'away_avg_conceded','ataque_vs_defensa_h','ataque_vs_defensa_a',
                  'over_proxy','b365h','b365d','b365a']
    X_goles = matches_sorted[feat_goles].dropna()
    y_goles = matches_sorted.loc[X_goles.index, 'total_goals']
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_goles, y_goles)

    return {
        'model_xg': model_xg, 'scaler': scaler,
        'features_xg': features_xg,
        'X_te_sc': X_te_sc, 'y_te': y_te,
        'y_pred_xg': y_pred_xg, 'y_prob_xg': y_prob_xg,
        'rf': rf, 'rf_probs': rf_probs, 'y_te_r': y_te_r,
        'log_match': log_match, 'feat_match': feat_match,
        'acc_cv': acc_cv, 'y_pred_match': y_pred_match,
        'X_log': X_log, 'y_log': y_log,
        'ridge': ridge, 'feat_goles': feat_goles,
        'matches_sorted': matches_sorted
    }

M = train_all_models(shots_final, matches)

# ─────────────────────────────────────────────
# HELPER: CANCHA
# ─────────────────────────────────────────────
def draw_pitch_opta(fig, bg="#0a1628"):
    lc = "rgba(255,255,255,0.25)"
    # Campo completo
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color=lc, width=1.5), fillcolor=bg)
    # Mediocampo
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color=lc, width=1.5))
    fig.add_shape(type="circle", x0=41, y0=41, x1=59, y1=59, line=dict(color=lc, width=1.5))
    # Área grande derecha
    fig.add_shape(type="rect", x0=83, y0=21.1, x1=100, y1=78.9, line=dict(color=lc, width=1.5))
    # Área chica derecha
    fig.add_shape(type="rect", x0=94, y0=36.8, x1=100, y1=63.2, line=dict(color=lc, width=1.5))
    # Portería derecha
    fig.add_shape(type="rect", x0=100, y0=45, x1=102, y1=55,
                  line=dict(color=lc, width=1.5), fillcolor=lc)
    # Área grande izquierda
    fig.add_shape(type="rect", x0=0, y0=21.1, x1=17, y1=78.9, line=dict(color=lc, width=1.5))
    fig.add_shape(type="rect", x0=0, y0=36.8, x1=6, y1=63.2, line=dict(color=lc, width=1.5))
    fig.add_shape(type="rect", x0=-2, y0=45, x1=0, y1=55,
                  line=dict(color=lc, width=1.5), fillcolor=lc)
    fig.update_xaxes(range=[-3, 103], showgrid=False, visible=False)
    fig.update_yaxes(range=[-3, 103], showgrid=False, visible=False)
    return fig

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("<div class='hero-title' style='font-size:3.2rem;'>Premier League ML Analytics</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Machine Learning I &mdash; Universidad Externado de Colombia &middot; 2026</div>", unsafe_allow_html=True)

# KPIs globales
k1, k2, k3, k4, k5 = st.columns(5)
total_shots = len(shots_final)
total_goals_shots = shots_final['is_goal'].sum()
conv_rate = total_goals_shots / total_shots * 100
auc_xg = roc_auc_score(M['y_te'], M['y_prob_xg'])
acc_match = M['acc_cv'].mean()

with k1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_shots:,}</div><div class='kpi-label'>Tiros Analizados</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_goals_shots:,}</div><div class='kpi-label'>Goles Registrados</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{conv_rate:.1f}%</div><div class='kpi-label'>Tasa Conversión</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{auc_xg:.3f}</div><div class='kpi-label'>AUC-ROC Modelo xG</div></div>", unsafe_allow_html=True)
with k5:
    color = "#4ade80" if acc_match > 0.498 else "#f87171"
    st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{color}'>{acc_match*100:.1f}%</div><div class='kpi-label'>Accuracy vs Bet365 (49.8%)</div></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='background:linear-gradient(135deg,#37003c,#1a0020);border:1px solid #00ff85;border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;'><div style='font-family:Bebas Neue,cursive;font-size:1.3rem;color:#00ff85;letter-spacing:3px;margin-bottom:0.8rem;'>CONTEXTO DEL PROYECTO</div><p style='color:#94a3b8;font-size:0.95rem;line-height:1.7;margin-bottom:0.8rem;'>La Premier League 2025/26 genera mas de <b style='color:white;'>444,000 eventos</b> por temporada. Cada tiro, pase y tackle queda registrado con coordenadas precisas. Las casas de apuestas como Bet365 aciertan el <b style='color:#00ff85;'>49.8%</b> de los partidos.</p><p style='color:#94a3b8;font-size:0.95rem;line-height:1.7;margin-bottom:0.8rem;'><b style='color:white;'>Nuestra mision:</b> construir un pipeline completo de ML que supere ese benchmark usando datos reales de <b style='color:white;'>7,198 tiros</b> y <b style='color:white;'>291 partidos</b>.</p><div style='display:flex;gap:1rem;margin-top:1rem;flex-wrap:wrap;'><div style='background:rgba(0,255,133,0.08);border:1px solid rgba(0,255,133,0.2);border-radius:8px;padding:0.6rem 1.2rem;'><div style='font-family:Bebas Neue,cursive;color:#00ff85;'>Modelo 1 - xG (Reg. Logistica)</div></div><div style='background:rgba(0,255,133,0.08);border:1px solid rgba(0,255,133,0.2);border-radius:8px;padding:0.6rem 1.2rem;'><div style='font-family:Bebas Neue,cursive;color:#00ff85;'>Modelo 2A - Goles Totales (Ridge)</div></div><div style='background:rgba(0,255,133,0.08);border:1px solid rgba(0,255,133,0.2);border-radius:8px;padding:0.6rem 1.2rem;'><div style='font-family:Bebas Neue,cursive;color:#00ff85;'>Modelo 2B - Resultado H/D/A</div></div><div style='background:rgba(0,255,133,0.08);border:1px solid rgba(0,255,133,0.2);border-radius:8px;padding:0.6rem 1.2rem;'><div style='font-family:Bebas Neue,cursive;color:#00ff85;'>Bonus - RF + XGBoost + K-Means</div></div></div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 EDA",
    "🗺️ Shot Map",
    "⚽ Modelo xG",
    "🏆 Match Predictor",
    "📈 Performance",
    "🧩 Clustering"
])

# ═══════════════════════════════════════════════
# TAB 1: EDA
# ═══════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-title'>Análisis Exploratorio de Datos</div>", unsafe_allow_html=True)
    st.markdown(r"""
    <div style="background:linear-gradient(135deg,#37003c,#1a0020);border:1px solid #00ff85;border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
    <div style="font-family:'Bebas Neue',cursive;font-size:1.3rem;color:#00ff85;letter-spacing:3px;margin-bottom:0.8rem;">CONTEXTO DEL PROYECTO</div>
    <p style="color:#94a3b8;font-size:0.95rem;line-height:1.7;margin-bottom:0.8rem;">
    La Premier League 2025/26 genera más de <b style="color:white;">444,000 eventos</b> por temporada. 
    Cada tiro, pase y tackle queda registrado con coordenadas precisas en una cancha normalizada de 100x100.
    Las casas de apuestas como Bet365 invierten millones en modelos predictivos y aciertan el 
    <b style="color:#00ff85;">49.8%</b> de los partidos.
    </p>
    <p style="color:#94a3b8;font-size:0.95rem;line-height:1.7;margin-bottom:0.8rem;">
    <b style="color:white;">Nuestra misión:</b> construir un pipeline completo de Machine Learning que intente superar ese benchmark 
    usando datos reales de <b style="color:white;">7,198 tiros</b> y <b style="color:white;">291 partidos</b> de la temporada actual.
    </p>
    <div style="display:flex;gap:2rem;margin-top:1rem;flex-wrap:wrap;">
         <div style="background:rgba(0,255,133,0.08);border:1px solid rgba(0,255,133,0.2);border-radius:8px;padding:0.6rem 1.2rem;">
              <div style="font-family:'Bebas Neue',cursive;color:#00ff85;font-size:1.1rem;">Modelo 1</div>
              <div style="color:#94a3b8;font-size:0.8rem;">xG - Expected Goals (Reg. Logística)</div>
         </div>
         <div style="background:rgba(0,255,133,0.08);border:1px solid rgba(0,255,133,0.2);border-radius:8px;padding:0.6rem 1.2rem;">
              <div style="font-family:'Bebas Neue',cursive;color:#00ff85;font-size:1.1rem;">Modelo 2A</div>
              <div style="color:#94a3b8;font-size:0.8rem;">Goles Totales (Ridge L2)</div>
         </div>
         <div style="background:rgba(0,255,133,0.08);border:1px solid rgba(0,255,133,0.2);border-radius:8px;padding:0.6rem 1.2rem;">
              <div style="font-family:'Bebas Neue',cursive;color:#00ff85;font-size:1.1rem;">Modelo 2B</div>
              <div style="color:#94a3b8;font-size:0.8rem;">Resultado H/D/A (Reg. Logística Multinomial)</div>
         </div>
         <div style="background:rgba(0,255,133,0.08);border:1px solid rgba(0,255,133,0.2);border-radius:8px;padding:0.6rem 1.2rem;">
              <div style="font-family:'Bebas Neue',cursive;color:#00ff85;font-size:1.1rem;">Bonus</div>
              <div style="color:#94a3b8;font-size:0.8rem;">Random Forest + XGBoost + K-Means</div>
         </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

   # BUSCA todo el bloque de fig1 y reemplaza por:
    with col1:
      n_gol = int(shots['is_goal'].sum())
      n_no_gol = int(len(shots) - n_gol)
      total = n_gol + n_no_gol

      fig1 = go.Figure(go.Bar(
        x=['No Gol', 'Gol'],
        y=[n_no_gol/total*100, n_gol/total*100],
        marker_color=['#37003c', '#00ff85'],
        text=[f'{n_no_gol/total*100:.1f}%', f'{n_gol/total*100:.1f}%'],
        textposition='outside',
        textfont=dict(color='white', size=16)
      ))
      fig1.update_layout(
        title='Desbalance de Clases — Variable Objetivo',
        title_font=dict(color='#00ff85', size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        xaxis=dict(color='white', showgrid=False),
        yaxis=dict(color='white', range=[0, 105],
                   gridcolor='rgba(255,255,255,0.1)')
      )
      st.plotly_chart(fig1, use_container_width=True)
      st.markdown("<div class='insight-box'>Solo el <b>11.2%</b> de los tiros son gol — desbalance 8:1. Un modelo naive que siempre prediga No Gol alcanza 88.8% de accuracy pero nunca detecta un gol real. Por esto usamos <b>AUC-ROC</b> y <b>class_weight=balanced</b>.</div>", unsafe_allow_html=True)
    with col2:
        bc_stats = shots_final.groupby('is_big_chance')['is_goal'].mean().reset_index()
        bc_stats['Tipo'] = bc_stats['is_big_chance'].map({0: 'Tiro Normal', 1: 'Big Chance'})
        bc_stats['Conversion'] = bc_stats['is_goal'] * 100

        fig2 = px.bar(bc_stats, x='Tipo', y='Conversion',
                      color='Tipo',
                      color_discrete_map={'Tiro Normal': '#37003c', 'Big Chance': '#00ff85'},
                      text=bc_stats['Conversion'].apply(lambda x: f'{x:.1f}%'),
                      title='Big Chance: El Predictor mas Poderoso')
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, font=dict(color='white'),
            title_font=dict(color='#00ff85'),
            xaxis=dict(color='white'), yaxis=dict(color='white', title='Conversion (%)'))
        fig2.update_traces(textposition='outside', textfont=dict(color='white', size=14))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div class='insight-box'>🎯 Un <b>Big Chance</b> tiene <b>7x mayor</b> probabilidad de gol (36.6% vs 5.5%). Es la variable con mayor <b>Information Gain</b> en el modelo xG — captura el contexto tactico (1v1, portero descolocado) que la geometria sola no puede medir.</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2) 
    with col3:
        bins = [0, 15, 30, 45, 60, 75, 90, 105]
        labels = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '90+']
        shots['intervalo'] = pd.cut(shots['minute'], bins=bins, labels=labels)

        tiempo_stats = shots.groupby('intervalo', observed=True)['is_goal'].agg(
            count='count', mean='mean').reset_index()
        tiempo_stats['conversion'] = tiempo_stats['mean'] * 100

        fig3 = go.Figure()
        fig3.add_bar(x=tiempo_stats['intervalo'].astype(str),
                     y=tiempo_stats['count'],
                     name='Tiros', marker_color='#37003c', yaxis='y')
        fig3.add_scatter(x=tiempo_stats['intervalo'].astype(str),
                         y=tiempo_stats['conversion'],
                         name='Conversion %',
                         line=dict(color='#00ff85', width=3),
                         marker=dict(size=10, color='#00ff85',
                                     line=dict(color='white', width=2)),
                         yaxis='y2', mode='lines+markers')
        fig3.update_layout(
            title='Volumen vs Efectividad por Intervalo',
            title_font=dict(color='#00ff85', size=14),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(title='Tiros', color='white',
                       gridcolor='rgba(255,255,255,0.08)'),
            yaxis2=dict(title='Conversion (%)', overlaying='y',
                        side='right', color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
            bargap=0.3
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("<div class='insight-box'>El minuto <b>90+</b> tiene la mayor conversion (>13%) con el menor volumen — fatiga defensiva. Justifica <code>is_final_minutes</code> como feature.</div>", unsafe_allow_html=True)

    with col4:
        # Distribución resultados H/D/A
        result_counts = matches['ftr'].value_counts()
        fig4 = px.pie(
            values=result_counts.values,
            names=['Home Win', 'Draw', 'Away Win'],
            color_discrete_sequence=['#00ff85', '#37003c', '#818cf8'],
            hole=0.55,
            title='Distribucion de Resultados'
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(color='#00ff85'),
            legend=dict(font=dict(color='white')))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("<div class='insight-box'>🏠 El local gana el <b>42.3%</b> de los partidos — la ventaja de localía es real. Los <b>empates (26.1%)</b> son el resultado mas difícil de predecir: el modelo logístico tiene Recall=0 para Draw, lo que refleja la alta aleatoriedad táctica de este resultado.</div>", unsafe_allow_html=True)

    # Top goleadores
st.markdown("<div class='section-title'>Top Goleadores</div>", unsafe_allow_html=True)
top10 = players.nlargest(10, 'goals_scored')[
        ['web_name','team','position','goals_scored','xG']].copy()
top10['xG'] = pd.to_numeric(top10['xG'], errors='coerce').round(2)
top10['Diferencia xG'] = (top10['goals_scored'] - top10['xG']).round(2)
top10 = top10.rename(columns={
        'web_name': 'Jugador', 'team': 'Equipo',
        'position': 'Posicion', 'goals_scored': 'Goles',
        'xG': 'xG Esperado'
})

def color_diff(val):
        if val > 0:
            return 'color: #00ff85; font-weight: bold'
        elif val < 0:
            return 'color: #f87171; font-weight: bold'
        return 'color: white'

styled = top10.style\
        .map(color_diff, subset=['Diferencia xG'])\
        .format({'xG Esperado': '{:.2f}', 'Diferencia xG': '{:.2f}'})\
        .hide(axis='index')\
        .set_properties(**{
            'background-color': '#0a0012',
            'color': 'white',
            'width': '100%'
        })\
        .set_table_styles([
            {'selector': 'table', 'props': [
                ('width', '100%'),
                ('border-collapse', 'collapse')
            ]},
            {'selector': 'th', 'props': [
                ('background-color', '#37003c'),
                ('color', '#00ff85'),
                ('font-weight', 'bold'),
                ('border', '1px solid #00ff85'),
                ('padding', '10px 16px'),
                ('text-align', 'left')
            ]},
            {'selector': 'td', 'props': [
                ('border', '1px solid rgba(0,255,133,0.15)'),
                ('padding', '10px 16px')
            ]}
        ])

    st.write(styled.to_html(), unsafe_allow_html=True)
# ═══════════════════════════════════════════════
# TAB 2: SHOT MAP
# ═══════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-title'>Mapa de Tiros con xG Predicho</div>", unsafe_allow_html=True)

    c_filter1, c_filter2, c_filter3 = st.columns(3)
    with c_filter1:
        filtro_tipo = st.selectbox("Mostrar", ["Todos los tiros", "Solo Goles", "Solo No Goles"])
    with c_filter2:
        filtro_bc = st.checkbox("Solo Big Chances")
    with c_filter3:
        filtro_equipo = st.selectbox("Equipo", ["Todos"] + sorted(shots_final['team_name'].dropna().unique().tolist()))

    # Calcular xG con el modelo
    features_xg = M['features_xg']
    X_all = shots_final[features_xg].fillna(0)
    shots_final['xg_pred'] = M['model_xg'].predict_proba(M['scaler'].transform(X_all))[:, 1]

    df_map = shots_final.copy()
    if filtro_tipo == "Solo Goles":
        df_map = df_map[df_map['is_goal'] == 1]
    elif filtro_tipo == "Solo No Goles":
        df_map = df_map[df_map['is_goal'] == 0]
    if filtro_bc:
        df_map = df_map[df_map['is_big_chance'] == 1]
    if filtro_equipo != "Todos":
        df_map = df_map[df_map['team_name'] == filtro_equipo]

    df_map = df_map[df_map['x'].between(0, 100) & df_map['y'].between(0, 100)]

    fig_map = px.scatter(
        df_map, x='x', y='y',
        color='xg_pred',
        color_continuous_scale='RdYlGn',
        size='xg_pred',
        size_max=20,
        opacity=0.75,
        hover_data={'x': False, 'y': False,
                    'xg_pred': ':.3f',
                    'minute': True,
                    'team_name': True,
                    'is_goal': True,
                    'is_big_chance': True},
        labels={'xg_pred': 'xG', 'minute': 'Minuto', 'team_name': 'Equipo', 'is_goal': 'Gol'},
        title=f'Shot Map — {filtro_equipo} ({len(df_map):,} tiros)'
    )
    fig_map = draw_pitch_opta(fig_map, bg="#0a1628")
    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0a1628',
        height=600,
        coloraxis_colorbar=dict(title='xG Predicho', tickfont=dict(color='#94a3b8')),
        font=dict(color='#94a3b8'),
        margin=dict(t=40, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Tiros Mostrados", f"{len(df_map):,}")
    with m2:
        st.metric("Goles", f"{df_map['is_goal'].sum():,}")
    with m3:
        st.metric("xG Promedio Predicho", f"{df_map['xg_pred'].mean():.3f}")

# ═══════════════════════════════════════════════
# TAB 3: MODELO xG INTERACTIVO
# ═══════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-title'>Modelo xG — Calculadora Interactiva</div>", unsafe_allow_html=True)

    col_inp, col_out = st.columns([1, 1.5])

    with col_inp:
        st.markdown("#### Configura el Tiro")
        angulo_input = st.slider("Ángulo de visión al arco (°)", 0.0, 90.0, 25.0, 0.5)
        zona = 0
        if angulo_input <= 11: zona = 0
        elif angulo_input <= 28: zona = 1
        elif angulo_input <= 57: zona = 2
        elif angulo_input <= 86: zona = 3
        else: zona = 4

        zona_nombres = {0: "Zona 0 — Casi imposible", 1: "Zona 1 — Costado área",
                        2: "Zona 2 — Borde área", 3: "Zona 3 — Interior área",
                        4: "Zona 4 — Frente al arco"}
        st.info(f"📍 {zona_nombres[zona]}")

        is_bc  = st.checkbox("¿Big Chance?", value=False)
        is_pen = st.checkbox("¿Penal?", value=False)
        is_hd  = st.checkbox("¿Cabezazo?", value=False)
        is_fb  = st.checkbox("¿Contraataque?", value=False)
        is_ft  = st.checkbox("¿Primer toque?", value=False)
        is_ag  = st.checkbox("¿Dentro del área grande?", value=True)
        is_ac  = st.checkbox("¿Dentro del área chica?", value=False)
        is_fm  = st.checkbox("¿Minuto 85+?", value=False)

        x_pred_raw = {
            'angulo_zona': zona,
            'is_big_chance': int(is_bc),
            'is_penalty': int(is_pen),
            'is_header': int(is_hd),
            'is_fast_break': int(is_fb),
            'first_touch': int(is_ft),
            'en_area_grande': int(is_ag),
            'en_area_chica': int(is_ac),
            'is_final_minutes': int(is_fm)
        }
        if 'goal_mouth_z' in M['features_xg']:
            x_pred_raw['goal_mouth_z'] = 1.5

        X_pred = pd.DataFrame([x_pred_raw])[M['features_xg']]
        X_pred_sc = M['scaler'].transform(X_pred)
        xg_prob = M['model_xg'].predict_proba(X_pred_sc)[0][1]

    with col_out:
        # Gauge
        color_gauge = "#4ade80" if xg_prob > 0.3 else "#facc15" if xg_prob > 0.1 else "#f87171"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=xg_prob * 100,
            number={'suffix': "%", 'font': {'size': 72, 'color': color_gauge,
                                             'family': 'Bebas Neue'}},
            title={'text': "Expected Goals (xG)", 'font': {'size': 18, 'color': '#94a3b8'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#475569'},
                'bar': {'color': color_gauge, 'thickness': 0.25},
                'bgcolor': '#0f172a',
                'bordercolor': '#1e293b',
                'steps': [
                    {'range': [0, 15],  'color': 'rgba(239,68,68,0.15)'},
                    {'range': [15, 40], 'color': 'rgba(250,204,21,0.15)'},
                    {'range': [40, 100],'color': 'rgba(74,222,128,0.15)'}
                ],
                'threshold': {'line': {'color': 'white', 'width': 3}, 'value': 11.2}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=320,
                                 margin=dict(t=20, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Referencia
        refs = {'Tiro Promedio': 0.112, 'Big Chance': 0.366, 'Penal': 0.829, 'Este Tiro': xg_prob}
        fig_ref = px.bar(
            x=list(refs.keys()), y=[v*100 for v in refs.values()],
            color=list(refs.keys()),
            color_discrete_map={'Tiro Promedio':'#1e3a5f','Big Chance':'#818cf8',
                                 'Penal':'#f472b6','Este Tiro':color_gauge},
            text=[f'{v*100:.1f}%' for v in refs.values()],
            title='Comparación con Benchmarks'
        )
        fig_ref.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               showlegend=False, font=dict(color='#94a3b8'),
                               yaxis_title='xG (%)')
        fig_ref.update_traces(textposition='outside')
        st.plotly_chart(fig_ref, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 4: MATCH PREDICTOR
# ═══════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-title'>Match Predictor — ¿Quién Gana?</div>", unsafe_allow_html=True)

    st.info("Selecciona las cuotas pre-partido de Bet365 para predecir el resultado H/D/A usando el modelo de regresión logística multinomial entrenado con datos reales.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Cuotas Bet365")
        cuota_h = st.number_input("Cuota Victoria Local (H)", min_value=1.01, max_value=20.0, value=2.10, step=0.05)
        cuota_d = st.number_input("Cuota Empate (D)", min_value=1.01, max_value=20.0, value=3.40, step=0.05)
        cuota_a = st.number_input("Cuota Victoria Visitante (A)", min_value=1.01, max_value=20.0, value=3.60, step=0.05)

        # Probabilidades implícitas
        inv_h, inv_d, inv_a = 1/cuota_h, 1/cuota_d, 1/cuota_a
        total_inv = inv_h + inv_d + inv_a
        imp_h = inv_h / total_inv
        imp_d = inv_d / total_inv
        imp_a = inv_a / total_inv

        X_match_pred = pd.DataFrame([[cuota_h, cuota_d, cuota_a, imp_h, imp_d, imp_a]],
                                     columns=M['feat_match'])
        probs = M['log_match'].predict_proba(X_match_pred)[0]
        classes = M['log_match'].classes_
        prob_dict = dict(zip(classes, probs))

        pred_result = classes[np.argmax(probs)]
        resultado_labels = {'H': '🏠 Victoria Local', 'D': '🤝 Empate', 'A': '✈️ Victoria Visitante'}

    with col_b:
        st.markdown("#### Predicción del Modelo")
        st.markdown(f"<div style='background:#0f172a;border:2px solid #38bdf8;border-radius:12px;padding:1.5rem;text-align:center;'>"
                    f"<div style='font-size:0.8rem;letter-spacing:2px;color:#64748b;'>RESULTADO PREDICHO</div>"
                    f"<div style='font-family:Bebas Neue;font-size:2.5rem;color:#38bdf8;margin:0.5rem 0;'>"
                    f"{resultado_labels.get(pred_result, pred_result)}</div>"
                    f"<div style='color:#94a3b8;font-size:0.9rem;'>Confianza: {max(probs)*100:.1f}%</div>"
                    f"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probabilidades como barras
        labels_map = {'H': 'Local', 'D': 'Empate', 'A': 'Visitante'}
        colors_map = {'H': '#38bdf8', 'D': '#64748b', 'A': '#818cf8'}

        for cls in ['H', 'D', 'A']:
            p = prob_dict.get(cls, 0)
            st.markdown(
                f"<div style='margin:0.3rem 0;'>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.85rem;color:#94a3b8;'>"
                f"<span>{labels_map[cls]}</span><span>{p*100:.1f}%</span></div>"
                f"<div style='background:#1e293b;border-radius:4px;height:8px;margin-top:3px;'>"
                f"<div style='background:{colors_map[cls]};width:{p*100:.1f}%;height:8px;border-radius:4px;'></div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

    # Goles esperados con Ridge
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Goles Esperados (Modelo Ridge + Historial)")

    ms = M['matches_sorted']
    equipos = sorted(matches['home_team'].dropna().unique())

    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        equipo_local = st.selectbox("Equipo Local", equipos, index=0)
    with col_eq2:
        equipo_visit = st.selectbox("Equipo Visitante", equipos, index=1)

    # Promedios históricos del equipo
    hist_local = ms[ms['home_team'] == equipo_local][['home_avg_scored','home_avg_conceded']].dropna().tail(1)
    hist_visit = ms[ms['away_team'] == equipo_visit][['away_avg_scored','away_avg_conceded']].dropna().tail(1)

    if not hist_local.empty and not hist_visit.empty:
        hs = hist_local['home_avg_scored'].values[0]
        hc = hist_local['home_avg_conceded'].values[0]
        as_ = hist_visit['away_avg_scored'].values[0]
        ac = hist_visit['away_avg_conceded'].values[0]
        ataque_h = hs + ac
        ataque_a = as_ + hc
        over_proxy = 1/cuota_h + 1/cuota_a

        X_goles_pred = pd.DataFrame([[hs, hc, as_, ac, ataque_h, ataque_a,
                                       over_proxy, cuota_h, cuota_d, cuota_a]],
                                     columns=M['feat_goles'])
        goles_pred = M['ridge'].predict(X_goles_pred)[0]

        g1, g2, g3 = st.columns(3)
        with g1:
            st.metric("Goles Esperados Totales", f"{goles_pred:.2f}")
        with g2:
            st.metric(f"Forma ofensiva {equipo_local}", f"{hs:.2f} goles/partido")
        with g3:
            st.metric(f"Forma ofensiva {equipo_visit}", f"{as_:.2f} goles/partido")
    else:
        st.warning("No hay suficiente historial para estos equipos. Se necesitan al menos 2 partidos previos.")

# ═══════════════════════════════════════════════
# TAB 5: PERFORMANCE
# ═══════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-title'>Performance de los Modelos</div>", unsafe_allow_html=True)

    # — Benchmark Bet365 —
    st.markdown("#### Comparación vs Bet365 (Match Predictor)")
    BENCHMARK = 0.498
    acc_m = M['acc_cv'].mean()

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:#94a3b8'>{BENCHMARK*100:.1f}%</div><div class='kpi-label'>Bet365 Accuracy</div></div>", unsafe_allow_html=True)
    with b2:
        color_acc = "#4ade80" if acc_m > BENCHMARK else "#f87171"
        st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{color_acc}'>{acc_m*100:.2f}%</div><div class='kpi-label'>Nuestro Modelo (CV=5)</div></div>", unsafe_allow_html=True)
    with b3:
        diff = (acc_m - BENCHMARK)*100
        color_diff = "#4ade80" if diff > 0 else "#f87171"
        signo = "+" if diff > 0 else ""
        st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{color_diff}'>{signo}{diff:.2f}%</div><div class='kpi-label'>Diferencia</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        # Confusion matrix Match Predictor
        cm_match = confusion_matrix(M['y_log'], M['y_pred_match'], labels=['H','D','A'])
        fig_cm_match = px.imshow(
            cm_match,
            x=['Home','Draw','Away'], y=['Home','Draw','Away'],
            color_continuous_scale='Blues',
            text_auto=True,
            title=f'Matriz de Confusión — Match Predictor ({acc_m*100:.2f}%)'
        )
        fig_cm_match.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#94a3b8'))
        st.plotly_chart(fig_cm_match, use_container_width=True)

    with col_cm2:
        # Confusion matrix xG
        cm_xg = confusion_matrix(M['y_te'], M['y_pred_xg'])
        fig_cm_xg = px.imshow(
            cm_xg,
            x=['No Gol','Gol'], y=['No Gol','Gol'],
            color_continuous_scale='Blues',
            text_auto=True,
            title=f'Matriz de Confusión — Modelo xG'
        )
        fig_cm_xg.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#94a3b8'))
        st.plotly_chart(fig_cm_xg, use_container_width=True)

    # Curvas ROC
    st.markdown("#### Curvas ROC — Modelo xG")
    col_roc1, col_roc2 = st.columns(2)

    with col_roc1:
        fpr_l, tpr_l, _ = roc_curve(M['y_te'], M['y_prob_xg'])
        fpr_rf, tpr_rf, _ = roc_curve(M['y_te_r'], M['rf_probs'])
        auc_l = roc_auc_score(M['y_te'], M['y_prob_xg'])
        auc_rf = roc_auc_score(M['y_te_r'], M['rf_probs'])

        fig_roc = go.Figure()
        fig_roc.add_scatter(x=fpr_l, y=tpr_l, mode='lines',
                            name=f'Logística (AUC={auc_l:.4f})',
                            line=dict(color='#38bdf8', width=2.5))
        fig_roc.add_scatter(x=fpr_rf, y=tpr_rf, mode='lines',
                            name=f'Random Forest (AUC={auc_rf:.4f})',
                            line=dict(color='#4ade80', width=2.5))
        fig_roc.add_scatter(x=[0,1], y=[0,1], mode='lines',
                            name='Aleatorio (AUC=0.5)',
                            line=dict(color='#475569', width=1.5, dash='dash'))
        fig_roc.update_layout(
            title='Curvas ROC — Logística vs Random Forest',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis_title='Tasa de Falsos Positivos',
            yaxis_title='Tasa de Verdaderos Positivos',
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_roc2:
        # Métricas resumidas
        from sklearn.metrics import precision_score, recall_score, f1_score
        metrics_data = {
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Logística xG': [
                f"{accuracy_score(M['y_te'], M['y_pred_xg'])*100:.2f}%",
                f"{precision_score(M['y_te'], M['y_pred_xg'])*100:.2f}%",
                f"{recall_score(M['y_te'], M['y_pred_xg'])*100:.2f}%",
                f"{f1_score(M['y_te'], M['y_pred_xg'])*100:.2f}%",
                f"{auc_l:.4f}"
            ],
            'Baseline Naive': ['88.80%', '0.00%', '0.00%', '0.00%', '0.5000']
        }
        st.markdown("#### Métricas Modelo xG vs Baseline")
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

        st.markdown("<div class='insight-box'>El baseline naive (siempre predice 'No Gol') tiene 88.8% de accuracy pero <b>nunca detecta un gol</b>. El AUC-ROC mide la capacidad real de discriminación independientemente del umbral.</div>", unsafe_allow_html=True)

        # Accuracy por fold
        st.markdown("#### Estabilidad por Fold (Match Predictor)")
        fold_df = pd.DataFrame({'Fold': [f'Fold {i+1}' for i in range(5)],
                                 'Accuracy': M['acc_cv'] * 100,
                                 'Bet365': [49.8]*5})
        fig_fold = go.Figure()
        fig_fold.add_bar(x=fold_df['Fold'], y=fold_df['Accuracy'],
                          name='Nuestro Modelo', marker_color='#38bdf8')
        fig_fold.add_scatter(x=fold_df['Fold'], y=fold_df['Bet365'],
                              name='Bet365 (49.8%)', mode='lines',
                              line=dict(color='#f87171', width=2, dash='dash'))
        fig_fold.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#94a3b8'),
                                legend=dict(bgcolor='rgba(0,0,0,0)'),
                                yaxis=dict(range=[30, 70]))
        st.plotly_chart(fig_fold, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 6: CLUSTERING
# ═══════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-title'>Clustering de Tiros — K-Means</div>", unsafe_allow_html=True)
    st.write("Segmentación no supervisada de tiros por zona, ángulo y tipo de ocasión.")

    col_k1, col_k2 = st.columns([1, 2])

    with col_k1:
        k = st.slider("Número de clusters (K)", 2, 6, 3)
        feat_km = ['distancia', 'angulo_grados', 'is_big_chance', 'en_area_grande']
        X_km = shots_final[feat_km].fillna(0)
        scaler_km = StandardScaler()
        X_km_sc = scaler_km.fit_transform(X_km)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        shots_final['cluster'] = km.fit_predict(X_km_sc).astype(str)

        # Perfil de clusters
        profile = shots_final.groupby('cluster').agg(
            Tiros=('is_goal', 'count'),
            Goles=('is_goal', 'sum'),
            Distancia_Media=('distancia', 'mean'),
            Angulo_Medio=('angulo_grados', 'mean'),
            BigChance_Rate=('is_big_chance', 'mean')
        ).reset_index()
        profile['Conversión (%)'] = (profile['Goles'] / profile['Tiros'] * 100).round(1)
        profile['Distancia_Media'] = profile['Distancia_Media'].round(1)
        profile['Angulo_Medio'] = profile['Angulo_Medio'].round(1)
        profile['BigChance_Rate'] = (profile['BigChance_Rate'] * 100).round(1)

        st.markdown("**Perfil de Clusters:**")
        st.dataframe(profile[['cluster','Tiros','Conversión (%)','Distancia_Media',
                               'Angulo_Medio','BigChance_Rate']].rename(
            columns={'cluster':'Cluster','Distancia_Media':'Dist. Media',
                     'Angulo_Medio':'Ángulo Medio','BigChance_Rate':'Big Chance %'}),
            use_container_width=True, hide_index=True)

    with col_k2:
        df_plot = shots_final[shots_final['x'].between(0,100) & shots_final['y'].between(0,100)].copy()
        fig_km = px.scatter(
            df_plot, x='x', y='y',
            color='cluster',
            opacity=0.6,
            size='distancia',
            size_max=12,
            color_discrete_sequence=px.colors.qualitative.Bold,
            title=f'Segmentación de Tiros — {k} Clusters'
        )
        fig_km = draw_pitch_opta(fig_km)
        fig_km.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0a1628',
            height=500,
            font=dict(color='#94a3b8'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=40, b=10)
        )
        st.plotly_chart(fig_km, use_container_width=True)

    st.markdown("<div class='insight-box'>El clustering revela patrones naturales en los tipos de tiro: tiros de larga distancia con bajo xG, remates en el área chica con alta conversión, y Big Chances distribuidas en el centro del área grande.</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:#1e293b;font-size:0.75rem;letter-spacing:2px;'>MACHINE LEARNING I · UNIVERSIDAD EXTERNADO DE COLOMBIA · 2026</p>", unsafe_allow_html=True)
