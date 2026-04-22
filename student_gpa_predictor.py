import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPA Predictor",
    page_icon="🎓",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0d1b2a 100%);
    color: #e8e8f0;
  }

  /* Title area */
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 300;
    letter-spacing: 0.04em;
  }
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
    border-left: 4px solid #a78bfa;
    padding-left: 0.7rem;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
  }

  /* Result card */
  .result-card {
    background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(96,165,250,0.1));
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 0 8px 32px rgba(167,139,250,0.12);
  }
  .gpa-value {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .grade-badge {
    display: inline-block;
    padding: 0.3rem 1.2rem;
    border-radius: 99px;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 0.6rem;
    letter-spacing: 0.05em;
  }

  /* Metric boxes */
  .metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    text-align: center;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
  }
  .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(167,139,250,0.2);
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stSlider > div > div > div { background: #a78bfa !important; }

  /* Divider */
  hr { border-color: rgba(255,255,255,0.08) !important; }

  /* Info tip */
  .tip-box {
    background: rgba(52,211,153,0.08);
    border-left: 3px solid #34d399;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    font-size: 0.88rem;
    color: #6ee7b7;
    margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Training Data (synthetic but realistic) ───────────────────────────────────
np.random.seed(42)
n = 200

study_hours  = np.random.uniform(1, 10, n)
screen_time  = np.random.uniform(1, 12, n)
noise        = np.random.normal(0, 0.18, n)

# GPA formula: study helps, screen hurts
gpa_raw = 1.5 + 0.28 * study_hours - 0.09 * screen_time + noise
gpa     = np.clip(gpa_raw, 0.0, 4.0)

X = np.column_stack([study_hours, screen_time])
y = gpa

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

train_preds = model.predict(X_scaled)
ss_res = np.sum((y - train_preds) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(np.mean((y - train_preds) ** 2))


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Student Input")
    st.markdown("---")

    st.markdown("**📚 Study Hours per Day**")
    input_study = st.slider(
        "study_hours", min_value=0.5, max_value=12.0,
        value=5.0, step=0.5, label_visibility="collapsed"
    )

    st.markdown("**📱 Screen Time per Day**")
    input_screen = st.slider(
        "screen_time", min_value=0.5, max_value=14.0,
        value=4.0, step=0.5, label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**📊 About the Model**")
    st.markdown(f"""
    <div style='font-size:0.83rem; color:#94a3b8; line-height:1.7'>
    • Algorithm: <b style='color:#a78bfa'>Linear Regression</b><br>
    • Training samples: <b style='color:#60a5fa'>200</b><br>
    • Features: Study Hours, Screen Time<br>
    • R² Score: <b style='color:#34d399'>{r2:.3f}</b><br>
    • RMSE: <b style='color:#f472b6'>{rmse:.3f}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔬 Built with Scikit-learn & Streamlit")


# ── Main Layout ────────────────────────────────────────────────────────────────
# Hero
st.markdown('<div class="hero-title">🎓 Student GPA Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Machine Learning · Linear Regression · Beginner Project</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Subheader / description
st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
    <div class="metric-box">
      <div class="metric-label">Step 1</div>
      <div class="metric-val">📥 Input</div>
      <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.4rem">Adjust sliders in the sidebar</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class="metric-box">
      <div class="metric-label">Step 2</div>
      <div class="metric-val">⚙️ Model</div>
      <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.4rem">Linear Regression computes GPA</div>
    </div>""", unsafe_allow_html=True)
with col_c:
    st.markdown("""
    <div class="metric-box">
      <div class="metric-label">Step 3</div>
      <div class="metric-val">📈 Result</div>
      <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.4rem">Your predicted GPA & letter grade</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ── Prediction ────────────────────────────────────────────────────────────────
input_arr    = np.array([[input_study, input_screen]])
input_scaled = scaler.transform(input_arr)
predicted_gpa = float(np.clip(model.predict(input_scaled)[0], 0.0, 4.0))

# Letter grade
def get_grade(gpa):
    if gpa >= 3.7: return "A",  "#34d399", "rgba(52,211,153,0.18)"
    if gpa >= 3.3: return "A−", "#6ee7b7", "rgba(110,231,183,0.15)"
    if gpa >= 3.0: return "B+", "#60a5fa", "rgba(96,165,250,0.18)"
    if gpa >= 2.7: return "B",  "#93c5fd", "rgba(147,197,253,0.15)"
    if gpa >= 2.3: return "B−", "#fbbf24", "rgba(251,191,36,0.15)"
    if gpa >= 2.0: return "C+", "#f97316", "rgba(249,115,22,0.15)"
    if gpa >= 1.7: return "C",  "#fb923c", "rgba(251,146,60,0.13)"
    if gpa >= 1.0: return "D",  "#f87171", "rgba(248,113,113,0.13)"
    return "F", "#ef4444", "rgba(239,68,68,0.12)"

letter, color, bg = get_grade(predicted_gpa)

left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.markdown('<div class="section-header">🔮 Prediction Result</div>', unsafe_allow_html=True)

    # Input summary
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-label">📚 Study Hours</div>
          <div class="metric-val">{input_study:.1f} hrs</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-label">📱 Screen Time</div>
          <div class="metric-val">{input_screen:.1f} hrs</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # GPA card
    st.markdown(f"""
    <div class="result-card">
      <div style="font-size:0.85rem;color:#94a3b8;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem">Predicted GPA</div>
      <div class="gpa-value">{predicted_gpa:.2f}</div>
      <div style="margin-top:0.7rem">
        <span class="grade-badge" style="background:{bg}; color:{color}; border:1px solid {color}40;">
          {letter}
        </span>
      </div>
      <div style="font-size:0.82rem;color:#64748b;margin-top:1rem">Scale: 0.00 – 4.00</div>
    </div>
    """, unsafe_allow_html=True)

    # Tip
    ratio = input_study / max(input_screen, 0.1)
    if ratio >= 1.2:
        tip = "✅ Great balance! High study-to-screen ratio supports academic success."
        tip_color = "#6ee7b7"; tip_border = "#34d399"
    elif ratio >= 0.7:
        tip = "⚠️ Consider reducing screen time or adding more study hours."
        tip_color = "#fcd34d"; tip_border = "#fbbf24"
    else:
        tip = "🚨 Screen time is significantly outweighing study hours — GPA impact is high."
        tip_color = "#fca5a5"; tip_border = "#ef4444"

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03); border-left:3px solid {tip_border};
         border-radius:0 10px 10px 0; padding:0.7rem 1rem; font-size:0.88rem;
         color:{tip_color}; margin-top:1rem">{tip}</div>
    """, unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="section-header">📊 Data Visualization</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.patch.set_facecolor("#0f0c29")

    for ax in axes:
        ax.set_facecolor("#13103a")
        ax.tick_params(colors="#64748b", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1b4b")

    # Plot 1 – Study hours vs GPA
    sc1 = axes[0].scatter(study_hours, gpa, c=gpa, cmap="cool", alpha=0.45, s=14, linewidths=0)
    axes[0].scatter([input_study], [predicted_gpa], color="#f472b6", s=90, zorder=5,
                    marker="*", label="You")
    axes[0].set_xlabel("Study Hours", color="#94a3b8", fontsize=8)
    axes[0].set_ylabel("GPA", color="#94a3b8", fontsize=8)
    axes[0].set_title("Study Hours vs GPA", color="#c4b5fd", fontsize=9, fontweight="bold")
    axes[0].legend(fontsize=7, facecolor="#1e1b4b", edgecolor="#312e81", labelcolor="white")

    # Plot 2 – Screen time vs GPA
    sc2 = axes[1].scatter(screen_time, gpa, c=gpa, cmap="cool", alpha=0.45, s=14, linewidths=0)
    axes[1].scatter([input_screen], [predicted_gpa], color="#f472b6", s=90, zorder=5,
                    marker="*", label="You")
    axes[1].set_xlabel("Screen Time", color="#94a3b8", fontsize=8)
    axes[1].set_ylabel("GPA", color="#94a3b8", fontsize=8)
    axes[1].set_title("Screen Time vs GPA", color="#c4b5fd", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=7, facecolor="#1e1b4b", edgecolor="#312e81", labelcolor="white")

    plt.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)

    # GPA range bar
    st.markdown("<br>", unsafe_allow_html=True)
    pct = predicted_gpa / 4.0
    bar_color = color

    st.markdown(f"""
    <div style="margin-top:0.5rem">
      <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#64748b;margin-bottom:4px">
        <span>0.0</span><span>GPA Scale</span><span>4.0</span>
      </div>
      <div style="background:#1e1b4b;border-radius:99px;height:12px;overflow:hidden;">
        <div style="width:{pct*100:.1f}%;height:100%;background:linear-gradient(90deg,{bar_color},{bar_color}99);
             border-radius:99px;transition:width 0.5s ease;"></div>
      </div>
      <div style="text-align:center;font-size:0.78rem;color:{color};margin-top:4px">
        {predicted_gpa:.2f} / 4.00
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Model Coefficients ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div class="section-header">🧠 Model Insights</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
coef_study  = model.coef_[0]
coef_screen = model.coef_[1]

for col, label, val, desc in [
    (c1, "Study Coefficient",  f"{coef_study:+.3f}",  "Each unit ↑ study → GPA change"),
    (c2, "Screen Coefficient", f"{coef_screen:+.3f}", "Each unit ↑ screen → GPA change"),
    (c3, "R² Score",           f"{r2:.3f}",            "Variance explained by model"),
    (c4, "RMSE",               f"{rmse:.3f}",          "Root mean squared error"),
]:
    col.markdown(f"""
    <div class="metric-box">
      <div class="metric-label">{label}</div>
      <div class="metric-val">{val}</div>
      <div style="font-size:0.75rem;color:#475569;margin-top:0.35rem">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("⚠️ This model uses synthetic training data for educational purposes. Results are illustrative, not clinically accurate.")
