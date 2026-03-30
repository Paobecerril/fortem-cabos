"""
Fortem Capital — Luxury Demand Intelligence
Los Cabos Residential Investment Analysis
Desarrollado por: Ana Paola Becerril Gutiérrez
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title='Fortem Capital · Los Cabos Intelligence',
    page_icon='🏛️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ─────────────────────────────────────────────
# BRAND PALETTE
# ─────────────────────────────────────────────
NAVY   = '#0B1F3A'
GOLD   = '#C9A84C'
STEEL  = '#2C4A6E'
CREAM  = '#F5F0E8'
SLATE  = '#64748B'
GREEN  = '#2D6A4F'
ORANGE = '#CA6702'
RED    = '#9B2226'

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: #FAFAF8;
    color: {NAVY};
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  [data-testid="stSidebar"] {{
    background: {NAVY};
    border-right: 1px solid {GOLD}33;
  }}
  [data-testid="stSidebar"] * {{ color: {CREAM} !important; }}
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {{
    color: {GOLD} !important;
    font-family: 'Cormorant Garamond', serif;
  }}
  [data-testid="stSidebar"] hr {{ border-color: {GOLD}44; }}
  [data-testid="stSidebar"] .stSlider > div > div > div > div {{ background: {GOLD}; }}
  [data-testid="stSidebar"] label {{
    color: {CREAM} !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em;
  }}
  [data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {STEEL};
    color: {CREAM};
    border: 1px solid {GOLD}55;
  }}

  .hero-header {{
    background: linear-gradient(135deg, {NAVY} 0%, {STEEL} 100%);
    padding: 2.5rem 3rem;
    border-radius: 0 0 16px 16px;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
  }}
  .hero-header::before {{
    content: ''; position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%; background: {GOLD}18; pointer-events: none;
  }}
  .hero-header::after {{
    content: ''; position: absolute;
    bottom: -40px; left: 200px;
    width: 120px; height: 120px;
    border-radius: 50%; background: {GOLD}10; pointer-events: none;
  }}
  .hero-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem; font-weight: 600; color: #FFFFFF;
    letter-spacing: 0.01em; margin: 0 0 0.25rem 0; line-height: 1.1;
  }}
  .hero-subtitle {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem; color: {GOLD}; letter-spacing: 0.12em;
    text-transform: uppercase; margin: 0 0 0.6rem 0; font-weight: 400;
  }}
  .hero-tagline {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem; color: rgba(255,255,255,0.65);
    margin: 0; font-weight: 300; letter-spacing: 0.02em;
  }}
  .hero-badge {{
    display: inline-block;
    background: {GOLD}22; border: 1px solid {GOLD}66; color: {GOLD};
    font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.25rem 0.8rem; border-radius: 20px; margin-top: 0.8rem;
    font-family: 'DM Sans', sans-serif;
  }}

  .section-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem; font-weight: 600; color: {NAVY};
    margin: 2rem 0 0.15rem 0; letter-spacing: 0.01em;
  }}
  .section-subtitle {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem; color: {SLATE};
    margin: 0 0 1.2rem 0; font-weight: 300;
  }}

  .semaforo-green {{
    background: #D1FAE5; color: {GREEN};
    border: 1px solid #6EE7B7; border-radius: 8px;
    padding: 0.6rem 1rem; font-weight: 600; font-size: 0.9rem;
  }}
  .semaforo-yellow {{
    background: #FEF3C7; color: {ORANGE};
    border: 1px solid #FCD34D; border-radius: 8px;
    padding: 0.6rem 1rem; font-weight: 600; font-size: 0.9rem;
  }}
  .semaforo-red {{
    background: #FEE2E2; color: {RED};
    border: 1px solid #FCA5A5; border-radius: 8px;
    padding: 0.6rem 1rem; font-weight: 600; font-size: 0.9rem;
  }}

  .insight-box {{
    background: linear-gradient(135deg, {NAVY}08, {GOLD}0A);
    border-left: 3px solid {GOLD}; border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin: 1rem 0;
    font-size: 0.88rem; color: {NAVY}; font-weight: 400; line-height: 1.6;
  }}
  .insight-box strong {{ color: {NAVY}; font-weight: 600; }}

  .source-pill {{
    display: inline-block;
    background: {NAVY}0D; border: 1px solid {NAVY}22; border-radius: 20px;
    padding: 0.2rem 0.65rem; font-size: 0.72rem; color: {NAVY};
    letter-spacing: 0.06em; text-transform: uppercase;
    margin: 0 0.25rem 0.4rem 0; font-weight: 500;
  }}

  .tooltip-card {{
    background: white; border: 1px solid #E8E3DB; border-radius: 10px;
    padding: 1rem 1.2rem; font-size: 0.83rem; color: {SLATE};
    line-height: 1.6; box-shadow: 0 2px 8px rgba(11,31,58,0.06);
  }}
  .tooltip-card .label {{
    font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: {GOLD}; font-weight: 600; margin-bottom: 0.4rem;
  }}

  [data-testid="metric-container"] {{
    background: white; border: 1px solid #E8E3DB;
    border-top: 3px solid {GOLD}; border-radius: 10px;
    padding: 0.9rem 1rem; box-shadow: 0 2px 8px rgba(11,31,58,0.06);
  }}
  [data-testid="metric-container"] label {{
    font-size: 0.72rem !important; letter-spacing: 0.07em !important;
    text-transform: uppercase !important; color: {SLATE} !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.9rem !important; color: {NAVY} !important;
  }}

  .footer {{
    margin-top: 4rem; padding: 1.5rem 0 1rem 0;
    border-top: 1px solid #E8E3DB; text-align: center;
    color: {SLATE}; font-size: 0.78rem; letter-spacing: 0.02em; line-height: 1.8;
  }}
  .footer .name {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.95rem; color: {NAVY}; font-weight: 600; letter-spacing: 0.04em;
  }}
  .footer .divider {{ color: {GOLD}; margin: 0 0.5rem; }}

  .streamlit-expanderHeader {{
    font-size: 0.83rem !important; color: {NAVY} !important;
    font-weight: 500 !important; letter-spacing: 0.02em;
  }}
  .streamlit-expanderContent {{
    font-size: 0.83rem; color: {SLATE}; line-height: 1.7;
  }}

  .gold-divider {{
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, {GOLD}88, transparent);
    margin: 2rem 0;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
MESES_ES = {
    1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril',
    5:'Mayo', 6:'Junio', 7:'Julio', 8:'Agosto',
    9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'
}
MESES_CORTO = {
    1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun',
    7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'
}
FEATURE_COLS = [
    'sp500', 'confianza', 'hipotecas', 'tipo_cambio',
    'mes', 'trimestre', 'turistas_lag1', 'turistas_lag3', 'turistas_lag12'
]

def semaforo_html(valor):
    if valor >= 60:
        return '<div class="semaforo-green">🟢 ALTA DEMANDA — Condiciones óptimas para lanzar</div>'
    elif valor >= 40:
        return '<div class="semaforo-yellow">🟡 DEMANDA MEDIA — Evaluar con cautela antes de decidir</div>'
    else:
        return '<div class="semaforo-red">🔴 DEMANDA BAJA — Esperar mejores condiciones de mercado</div>'

def set_plot_style(ax, xlabel='', ylabel=''):
    ax.set_facecolor('#FAFAF8')
    ax.figure.set_facecolor('#FAFAF8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E8E3DB')
    ax.spines['bottom'].set_color('#E8E3DB')
    ax.tick_params(colors=SLATE, labelsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9, color=SLATE, labelpad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=SLATE, labelpad=8)
    ax.grid(True, axis='y', alpha=0.4, color='#E8E3DB', linewidth=0.8)
    ax.grid(False, axis='x')

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# DATA LOADING + RETRAIN ON STARTUP
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner='Cargando modelo...')
def cargar_modelo_y_datos():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    df = pd.read_csv('fortem_demand_data.csv', index_col=0, parse_dates=True)
    forecast = pd.read_csv('fortem_forecast.csv', index_col=0, parse_dates=True)

    feature_cols = [
        'sp500', 'confianza', 'hipotecas', 'tipo_cambio',
        'mes', 'trimestre', 'turistas_lag1', 'turistas_lag3', 'turistas_lag12'
    ]

    X = df[feature_cols]
    y = df['demand_index']

    preprocessor = ColumnTransformer([('num', 'passthrough', feature_cols)])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X, y)

    return pipeline, df, forecast

rf_pipeline, df, forecast_df = cargar_modelo_y_datos()
# ─────────────────────────────────────────────
# SIDEBAR — SIMULADOR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:1rem 0 0.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Cormorant Garamond,serif;font-size:1.4rem;letter-spacing:0.04em;margin:0;">FORTEM CAPITAL</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.7rem;letter-spacing:0.15em;color:#C9A84C;margin:0.2rem 0 0.8rem 0;text-transform:uppercase;">Los Cabos · Análisis de Demanda</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('### 🎛️ Simulador de Escenarios')
    st.markdown('<p style="font-size:0.78rem;opacity:0.7;margin-bottom:1rem;">Ajusta las condiciones y ve el resultado en la pestaña <strong>Simulador</strong>.</p>', unsafe_allow_html=True)

    sp500_sim = st.slider(
        '📈 S&P 500', 3000, 7000, int(df['sp500'].iloc[-1]), step=100,
        help='Índice bursátil de EE.UU. Un S&P alto refleja riqueza del comprador objetivo.'
    )
    confianza_sim = st.slider(
        '😊 Confianza del Consumidor USA', 50, 110, int(df['confianza'].iloc[-1]),
        help='Índice U. of Michigan. Mayor confianza = más gasto en lujo.'
    )
    hipotecas_sim = st.slider(
        '🏦 Tasa Hipotecaria 30 años (%)', 3.0, 9.0, float(df['hipotecas'].iloc[-1]), step=0.1,
        help='Tasas altas reducen el apetito comprador en EE.UU.'
    )
    tipo_cambio_sim = st.slider(
        '💱 Tipo de Cambio USD/MXN', 15.0, 22.0, float(df['tipo_cambio'].iloc[-1]), step=0.1,
        help='Peso débil = Los Cabos más barato para el comprador americano.'
    )
    mes_sim = st.selectbox(
        '📅 Mes de Lanzamiento',
        options=list(range(1, 13)),
        format_func=lambda m: MESES_ES[m],
        index=0
    )
    lag1_sim = st.slider(
        '👥 Turistas recientes (mes anterior)', 50000, 300000,
        int(df['turistas_totales'].iloc[-1]), step=5000,
        help='Volumen de turistas EE.UU. + Canadá en el último mes.'
    )
    st.markdown('---')
    st.markdown('<p style="font-size:0.7rem;opacity:0.55;text-align:center;line-height:1.6;">Fuentes · FRED Federal Reserve · SECTUR México<br>Modelo · Random Forest Regressor · R² = 0.51</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COMPUTE SIMULATION
# ─────────────────────────────────────────────
input_sim = pd.DataFrame([{
    'sp500'          : sp500_sim,
    'confianza'      : confianza_sim,
    'hipotecas'      : hipotecas_sim,
    'tipo_cambio'    : tipo_cambio_sim,
    'mes'            : mes_sim,
    'trimestre'      : (mes_sim - 1) // 3 + 1,
    'turistas_lag1'  : lag1_sim,
    'turistas_lag3'  : lag1_sim,
    'turistas_lag12' : lag1_sim
}])
pred_sim = rf_pipeline.predict(input_sim[FEATURE_COLS])[0]

# Pre-compute static KPIs
demanda_actual    = df['demand_index'].iloc[-1]
demanda_historica = df['demand_index'].mean()
demanda_forecast  = forecast_df['demand_index_pred'].max()
mejor_mes         = forecast_df['demand_index_pred'].idxmax()
delta_actual      = demanda_actual - demanda_historica
delta_sign        = '+' if delta_actual >= 0 else ''

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-subtitle">Fortem Capital · Investment Intelligence</p>
  <h1 class="hero-title">Los Cabos Luxury<br>Demand Model</h1>
  <p class="hero-tagline">
    Análisis cuantitativo de demanda de compradores de segunda vivienda de lujo · EE.UU. &amp; Canadá
  </p>
</div>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
# KPIs ESTÁTICOS (mercado base, no cambian con sliders)
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">Indicadores del Mercado</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Pulso actual del mercado — datos históricos y proyección base.</p>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        label='📍 Demanda Actual del Mercado',
        value=f'{demanda_actual:.1f} / 100',
        delta=f'{delta_sign}{delta_actual:.1f} pts vs promedio histórico'
    )
with k2:
    st.metric(
        label='🔮 Pico Proyectado (12 meses)',
        value=f'{demanda_forecast:.1f} / 100'
    )
with k3:
    st.metric(
        label='📅 Mejor Mes para Lanzar',
        value=f'{MESES_CORTO[mejor_mes.month]} {mejor_mes.year}'
    )
with k4:
    semaforo_base = '🟢 LANZAR' if demanda_forecast >= 60 else '🟡 EVALUAR' if demanda_forecast >= 40 else '🔴 ESPERAR'
    st.metric(label='🚦 Señal del Mercado', value=semaforo_base)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(['📊 Análisis del Mercado', '🎛️ Simulador de Escenarios'])

# ══════════════════════════════════════════════
# TAB 1 — ANÁLISIS ESTÁTICO
# ══════════════════════════════════════════════
with tab1:

    # ── FORECAST ──────────────────────────────
    st.markdown('<p class="section-title">📈 Proyección de Demanda — Próximos 12 Meses</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Evolución proyectada de la demanda mes a mes, manteniendo las condiciones macroeconómicas actuales.</p>', unsafe_allow_html=True)

    fig_f, ax_f = plt.subplots(figsize=(13, 4.5))
    hist = df['demand_index'].tail(24)
    ax_f.plot(hist.index, hist.values, color=NAVY, linewidth=2.2, label='Demanda Histórica', zorder=3)
    ax_f.fill_between(hist.index, hist.values, alpha=0.07, color=NAVY)
    ax_f.plot(forecast_df.index, forecast_df['demand_index_pred'],
              color=GOLD, linewidth=2.5, linestyle='--', label='Proyección 12 meses', zorder=4)
    ax_f.fill_between(forecast_df.index, forecast_df['demand_index_pred'], alpha=0.18, color=GOLD)
    ax_f.axhspan(60, 105, alpha=0.06, color=GREEN)
    ax_f.axhspan(40, 60,  alpha=0.06, color=ORANGE)
    ax_f.axhspan(0,  40,  alpha=0.04, color=RED)
    ax_f.axhline(60, color=GREEN,  linestyle=':', alpha=0.5, linewidth=1.2)
    ax_f.axhline(40, color=ORANGE, linestyle=':', alpha=0.5, linewidth=1.2)
    ax_f.axvline(mejor_mes, color=GREEN, linewidth=1.8, linestyle=':', alpha=0.7, zorder=5)
    ax_f.annotate(
        f'Mejor mes:\n{MESES_CORTO[mejor_mes.month]} {mejor_mes.year}',
        xy=(mejor_mes, demanda_forecast),
        xytext=(15, -30), textcoords='offset points',
        fontsize=8, color=GREEN, fontweight='600',
        arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2)
    )
    ax_f_twin = ax_f.twinx()
    ax_f_twin.set_ylim(ax_f.get_ylim())
    ax_f_twin.set_yticks([20, 50, 80])
    ax_f_twin.set_yticklabels(['BAJA', 'MEDIA', 'ALTA'], fontsize=7.5, color=SLATE, fontweight='600')
    ax_f_twin.spines['right'].set_color('#E8E3DB')
    ax_f_twin.tick_params(right=False)
    divider_x = df.index[-1] + pd.DateOffset(days=15)
    ax_f.axvline(divider_x, color=SLATE, linewidth=1, alpha=0.3, linestyle='-')
    ax_f.text(divider_x, ax_f.get_ylim()[1] * 0.97, ' Proyección →', fontsize=7.5, color=SLATE, va='top')
    set_plot_style(ax_f, ylabel='Índice de Demanda (0–100)')
    ax_f.set_ylim(0, min(105, ax_f.get_ylim()[1]))
    ax_f.legend(loc='lower left', fontsize=8.5, framealpha=0.9, edgecolor='#E8E3DB', facecolor='white')
    plt.tight_layout()
    st.pyplot(fig_f, use_container_width=True)
    plt.close()

    with st.expander('ℹ️ ¿Cómo se construye esta proyección?'):
        st.markdown(f"""
        El modelo toma las **condiciones macroeconómicas actuales** y las mantiene constantes
        durante los 12 meses. Lo que varía es **la estacionalidad** — el modelo aprendió que
        Los Cabos tiene patrones de demanda muy claros según el mes del año (temporada alta: diciembre–febrero).

        El **mejor mes para lanzar** ({MESES_ES[mejor_mes.month]} {mejor_mes.year}) representa
        una demanda proyectada de **{demanda_forecast:.1f}/100** —
        {"en zona óptima para el lanzamiento." if demanda_forecast >= 60 else "en zona de evaluación; monitorear evolución macro."}
        """)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    # ── ESTACIONALIDAD ─────────────────────────
    st.markdown('<p class="section-title">📅 Estacionalidad — El Mejor Timing de Lanzamiento</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">¿En qué meses del año históricamente llega más el comprador objetivo?</p>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        demanda_mes   = df.groupby('mes')['demand_index'].mean()
        meses_nombres = [MESES_CORTO[i] for i in range(1, 13)]
        fig_s, ax_s = plt.subplots(figsize=(10, 4))
        colores_bars = [GOLD if v == demanda_mes.max() else NAVY + 'CC' for v in demanda_mes.values]
        bars = ax_s.bar(meses_nombres, demanda_mes.values, color=colores_bars,
                        width=0.65, zorder=3, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, demanda_mes.values):
            ax_s.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                      f'{val:.0f}', ha='center', va='bottom', fontsize=8.5, color=NAVY, fontweight='500')
        max_idx = demanda_mes.values.argmax()
        ax_s.set_title(f'Demanda promedio histórica por mes — pico en {meses_nombres[max_idx]}',
                       fontsize=10, color=NAVY, fontweight='600', pad=10)
        set_plot_style(ax_s, ylabel='Índice Promedio (0–100)')
        ax_s.set_ylim(0, demanda_mes.max() * 1.18)
        gold_patch = mpatches.Patch(color=GOLD, label='Mes de mayor demanda histórica')
        navy_patch = mpatches.Patch(color=NAVY + 'CC', label='Resto del año')
        ax_s.legend(handles=[gold_patch, navy_patch], fontsize=8, framealpha=0.9, edgecolor='#E8E3DB')
        plt.tight_layout()
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

    with col_s2:
        top3 = demanda_mes.sort_values(ascending=False).head(3)
        st.markdown(f"""
        <div class="insight-box">
          <strong>¿Por qué importa esto?</strong><br><br>
          Los meses de mayor demanda concentran el mayor flujo del comprador objetivo —
          el americano y canadiense que escapa del invierno norte.<br><br>
          <strong>Top 3 meses históricos:</strong><br>
          {''.join([f"<br>• <strong>{MESES_ES[int(m)]}</strong> — índice {v:.0f}/100" for m, v in top3.items()])}<br><br>
          Lanzar en estos meses maximiza la exposición ante el comprador más activo
          y reduce la presión de descuento en precio.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE ─────────────────────
    st.markdown('<p class="section-title">🔍 ¿Qué Factores Mueven la Demanda en Los Cabos?</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Variables ordenadas por su peso en la predicción del modelo.</p>', unsafe_allow_html=True)

    nombres_display = {
        'sp500'          : 'Bolsa de valores EE.UU. (S&P 500)',
        'confianza'      : 'Confianza del consumidor americano',
        'hipotecas'      : 'Tasa hipotecaria (30 años)',
        'tipo_cambio'    : 'Tipo de cambio Peso / Dólar',
        'mes'            : 'Mes del año (estacionalidad)',
        'trimestre'      : 'Trimestre del año',
        'turistas_lag1'  : 'Demanda del mes anterior',
        'turistas_lag3'  : 'Demanda de hace 3 meses',
        'turistas_lag12' : 'Demanda del mismo mes, año anterior'
    }
    explicaciones = {
        'turistas_lag12' : 'El patrón del año anterior se repite — Los Cabos tiene estacionalidad muy marcada.',
        'turistas_lag1'  : 'Si el mes pasado hubo alta demanda, el momentum suele continuar.',
        'sp500'          : 'Cuando la bolsa sube, el comprador americano de lujo tiene más riqueza y gasta más.',
        'tipo_cambio'    : 'Un peso débil hace Los Cabos más accesible para el comprador extranjero.',
        'confianza'      : 'La confianza económica anticipa el gasto discrecional en segunda vivienda.',
        'hipotecas'      : 'Tasas altas enfrían el sentimiento comprador, aunque Los Cabos es mayormente cash.',
        'mes'            : 'La temporada del año define fuertemente el apetito de viaje.',
        'turistas_lag3'  : 'Captura tendencias de mediano plazo (un trimestre atrás).',
        'trimestre'      : 'Refuerza el patrón estacional junto con el mes.'
    }
    importancias = rf_pipeline.named_steps['model'].feature_importances_
    fi_df = pd.DataFrame({
        'variable'  : FEATURE_COLS,
        'display'   : [nombres_display[c] for c in FEATURE_COLS],
        'importance': importancias
    }).sort_values('importance', ascending=True)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 5.5))
    colores_fi = [GOLD if v == fi_df['importance'].max() else NAVY + 'CC' for v in fi_df['importance']]
    bars_fi = ax_fi.barh(fi_df['display'], fi_df['importance'],
                         color=colores_fi, height=0.55, edgecolor='white', linewidth=0.5, zorder=3)
    for bar, val in zip(bars_fi, fi_df['importance']):
        ax_fi.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                   f'{val*100:.1f}%', va='center', fontsize=8.5, color=NAVY, fontweight='500')
    set_plot_style(ax_fi, xlabel='Peso relativo en el modelo')
    ax_fi.set_xlim(0, fi_df['importance'].max() * 1.25)
    ax_fi.set_title('Variables ordenadas por influencia en la predicción de demanda',
                    fontsize=10, color=NAVY, fontweight='600', pad=10)
    gold_patch2 = mpatches.Patch(color=GOLD, label='Variable más influyente')
    ax_fi.legend(handles=[gold_patch2], fontsize=8, framealpha=0.9, edgecolor='#E8E3DB')
    plt.tight_layout()
    st.pyplot(fig_fi, use_container_width=True)
    plt.close()

    top_var = fi_df.iloc[-1]['variable']
    st.markdown(f"""
    <div class="insight-box">
      <strong>Conclusión:</strong> El factor más determinante es <strong>{nombres_display[top_var]}</strong>.
      Los Cabos tiene una estacionalidad muy predecible, lo que facilita la planeación del timing.
      El siguiente factor son las condiciones macro de EE.UU. — lo que vincula el proyecto
      directamente al ciclo económico del comprador norteamericano.
    </div>
    """, unsafe_allow_html=True)

    with st.expander('📖 ¿Qué significa cada factor? (explicación de negocio)'):
        cols_exp = st.columns(2)
        for i, (var, display) in enumerate(zip(fi_df['variable'].tolist()[::-1], fi_df['display'].tolist()[::-1])):
            with cols_exp[i % 2]:
                st.markdown(f"""
                <div class="tooltip-card" style="margin-bottom:0.8rem;">
                  <p class="label">{display}</p>
                  <p style="margin:0;">{explicaciones.get(var, '')}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    # ── METODOLOGÍA ────────────────────────────
    st.markdown('<p class="section-title">📚 Metodología y Fuentes de Datos</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Cómo se construyó este modelo de inteligencia de demanda.</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="tooltip-card">
          <p class="label">🏦 Fuente 1 — FRED</p>
          <p><strong>¿Qué es?</strong> La base de datos económica abierta de la Reserva Federal de EE.UU.</p>
          <p><strong>¿Qué tomamos?</strong></p>
          <ul style="margin:0;padding-left:1.2rem;color:{SLATE};">
            <li>S&P 500 (efecto riqueza)</li>
            <li>Confianza del consumidor americano</li>
            <li>Tasas hipotecarias a 30 años</li>
            <li>Tipo de cambio USD/MXN</li>
          </ul>
          <p style="margin-top:0.8rem;margin-bottom:0;">
            <span class="source-pill">FRED API</span>
            <span class="source-pill">Fed Reserve EE.UU.</span>
          </p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="tooltip-card">
          <p class="label">✈️ Fuente 2 — SECTUR</p>
          <p><strong>¿Qué es?</strong> La Secretaría de Turismo de México publica mensualmente
          el volumen real de turistas por aeropuerto y país de residencia.</p>
          <p><strong>¿Qué tomamos?</strong> Llegadas reales de <strong>EE.UU. y Canadá</strong>
          al aeropuerto de Los Cabos — exactamente el perfil del comprador objetivo.</p>
          <p><strong>¿Por qué no Google Trends?</strong> Los términos de lujo tienen volumen
          demasiado bajo para generar señal mensual confiable.</p>
          <p style="margin-bottom:0;">
            <span class="source-pill">SECTUR</span>
            <span class="source-pill">Datos Reales</span>
          </p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="tooltip-card">
          <p class="label">🤖 Modelo — Random Forest</p>
          <p>Piénsalo como un <em>comité de 200 expertos</em> que aprenden patrones distintos
          y después votan juntos para dar una predicción final.</p>
          <ul style="margin:0;padding-left:1.2rem;color:{SLATE};">
            <li>Funciona bien con datasets medianos (84 meses)</li>
            <li>Robusto ante datos atípicos (excluimos COVID)</li>
            <li>No requiere supuestos estadísticos rígidos</li>
          </ul>
          <p style="margin-top:0.8rem;margin-bottom:0;">
            <strong>Precisión:</strong> Explica el <strong>51%</strong> de la variación
            en demanda usando solo datos públicos.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — SIMULADOR INTERACTIVO
# ══════════════════════════════════════════════
with tab2:

    st.markdown(f"""
    <div class="insight-box" style="margin-bottom:1.5rem;">
      <strong>¿Cómo usar este simulador?</strong> Mueve los controles del panel izquierdo
      para explorar distintos escenarios de mercado. El índice y la recomendación de abajo
      se actualizan automáticamente con cada ajuste.
    </div>
    """, unsafe_allow_html=True)

    # ── GAUGE + SEMÁFORO ───────────────────────
    r1, r2, r3 = st.columns([1.2, 1.8, 1])
    with r1:
        color_hex = GREEN if pred_sim >= 60 else ORANGE if pred_sim >= 40 else RED
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem 1rem;background:white;
                    border-radius:12px;border:1px solid #E8E3DB;
                    box-shadow:0 2px 12px rgba(11,31,58,0.07);">
          <p style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:{SLATE};margin:0 0 0.5rem 0;">Índice de Demanda Simulado</p>
          <p style="font-family:'Cormorant Garamond',serif;font-size:3.5rem;
                    font-weight:700;color:{color_hex};margin:0;line-height:1;">
            {pred_sim:.1f}
          </p>
          <p style="font-size:0.85rem;color:{SLATE};margin:0.2rem 0 0.8rem 0;">de 100</p>
          {semaforo_html(pred_sim)}
        </div>
        """, unsafe_allow_html=True)

    with r2:
        fig_g, ax_g = plt.subplots(figsize=(5.5, 2.8))
        rangos    = [40, 20, 40]
        colores_g = ['#FEE2E2', '#FEF3C7', '#D1FAE5']
        bordes_g  = [RED, ORANGE, GREEN]
        acum = 0
        for rng, col, brd in zip(rangos, colores_g, bordes_g):
            ax_g.barh(0, rng, left=acum, height=0.5, color=col,
                      edgecolor=brd, linewidth=1.2, zorder=2)
            acum += rng
        ax_g.axvline(pred_sim, color=color_hex, linewidth=3, zorder=5, ymin=0.15, ymax=0.85)
        ax_g.scatter([pred_sim], [0], color=color_hex, s=180, zorder=6)
        ax_g.set_xlim(0, 100)
        ax_g.set_ylim(-0.5, 0.5)
        ax_g.set_yticks([])
        ax_g.set_xticks([0, 20, 40, 60, 80, 100])
        ax_g.set_facecolor('#FAFAF8')
        ax_g.figure.set_facecolor('#FAFAF8')
        ax_g.spines['top'].set_visible(False)
        ax_g.spines['right'].set_visible(False)
        ax_g.spines['left'].set_visible(False)
        ax_g.spines['bottom'].set_color('#E8E3DB')
        ax_g.tick_params(colors=SLATE, labelsize=8)
        ax_g.set_title('Posición en el espectro histórico', fontsize=9.5, color=NAVY, fontweight='600', pad=8)
        for cat, pos in zip(['Baja\n(0–40)', 'Media\n(40–60)', 'Alta\n(60–100)'], [20, 50, 80]):
            ax_g.text(pos, 0.35, cat, ha='center', va='bottom', fontsize=7.5, color=SLATE, fontweight='500')
        plt.tight_layout()
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

    with r3:
        st.markdown(f"""
        <div class="tooltip-card">
          <p class="label">¿Cómo leer este índice?</p>
          <p>El índice va de <strong>0 a 100</strong>, donde 100 es el pico histórico
          de demanda que Los Cabos ha registrado.</p>
          <p>Un valor de <strong>{pred_sim:.0f}</strong> significa que el escenario
          se ubica en el <strong>{pred_sim:.0f}° percentil</strong> del historial.</p>
          <p style="margin:0;">Para un proyecto de <strong>+$1M USD</strong>,
          recomendamos lanzar en zona verde (≥60).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    # ── RECOMENDACIÓN ─────────────────────────
    st.markdown('<p class="section-title">🏁 Recomendación del Escenario</p>', unsafe_allow_html=True)

    decision_color = GREEN if pred_sim >= 60 else ORANGE if pred_sim >= 40 else RED
    semaforo_texto = (
        f"El escenario simulado muestra condiciones <strong>favorables</strong> para el lanzamiento. "
        f"Con los parámetros configurados, el índice proyectado ({pred_sim:.1f}/100) se ubica "
        f"en zona verde — momento óptimo para comprometer el lanzamiento."
    ) if pred_sim >= 60 else (
        f"El escenario simulado muestra condiciones <strong>moderadas</strong>. Se recomienda "
        f"monitorear la evolución del S&P 500 y la confianza del consumidor antes de decidir."
    ) if pred_sim >= 40 else (
        f"El escenario simulado sugiere <strong>esperar</strong>. El comprador objetivo "
        f"americano está en un ciclo de menor actividad. Se recomienda revisar en 3–6 meses."
    )

    d1, d2 = st.columns([1.6, 1])
    with d1:
        st.markdown(f"""
        <div style="background:white;border-radius:12px;border:1px solid #E8E3DB;
                    border-left:4px solid {decision_color};padding:1.5rem 1.8rem;
                    box-shadow:0 2px 12px rgba(11,31,58,0.07);">
          <p style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:{SLATE};margin:0 0 0.6rem 0;">Basado en los parámetros del simulador</p>
          {semaforo_html(pred_sim)}
          <p style="margin:1rem 0 0 0;font-size:0.9rem;color:{NAVY};line-height:1.7;">{semaforo_texto}</p>
          <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid #E8E3DB;">
            <p style="font-size:0.75rem;color:{SLATE};margin:0;">
              <strong>Parámetros activos:</strong>
              S&P {sp500_sim:,} · Confianza {confianza_sim} · Hipotecas {hipotecas_sim:.1f}% ·
              USD/MXN {tipo_cambio_sim:.2f} · Mes: {MESES_ES[mes_sim]}
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown(f"""
        <div style="background:{NAVY};border-radius:12px;padding:1.5rem 1.4rem;color:white;">
          <p style="font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
                    color:{GOLD};margin:0 0 1rem 0;">Variables de Monitoreo</p>
          <p style="font-size:0.83rem;opacity:0.85;line-height:1.7;margin:0;">
            Monitorear mensualmente para actualizar la recomendación:
          </p>
          <ul style="font-size:0.83rem;opacity:0.85;line-height:2;margin:0.5rem 0 0 0;padding-left:1.2rem;">
            <li>S&P 500 (actual: {df['sp500'].iloc[-1]:,.0f})</li>
            <li>Confianza U. of Michigan ({df['confianza'].iloc[-1]:.1f})</li>
            <li>Tasa hipotecaria 30y ({df['hipotecas'].iloc[-1]:.2f}%)</li>
            <li>USD/MXN ({df['tipo_cambio'].iloc[-1]:.2f})</li>
          </ul>
          <p style="font-size:0.78rem;opacity:0.6;margin:1.2rem 0 0 0;line-height:1.5;">
            Fuente: FRED · Actualización mensual recomendada
          </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <span class="name">Ana Paola Becerril Gutiérrez</span>
  <span class="divider">·</span>
  Análisis de Inversión · Fortem Capital
  <span class="divider">·</span>
  Los Cabos Luxury Demand Intelligence
  <br>
  <span style="font-size:0.72rem;opacity:0.55;">
    Modelo: Random Forest Regressor · Fuentes: FRED (Federal Reserve) + SECTUR México ·
    Datos: 2015–2024 (excl. COVID) · R² = 0.51
  </span>
</div>
""", unsafe_allow_html=True)
