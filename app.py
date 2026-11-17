import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from diabetes_ga_bpn import BPN, load_data, ZERO_COLS
import fuzzy_layer as fz
from app_content import (read_text, split_sections, OVERVIEW_MD, PIPELINE_DOT,
                         OVERVIEW_TAIL_MD, FEATURES, DATA_MD, NN_MD, GA_MD, FUZZY_MD,
                         AUG_MD, REFERENCES_MD)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="GlycoSense | Risk Assessment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    # Prefer the GA + BPN + Fuzzy hybrid (made by hybrid_experiment.py);
    # fall back to the original GA-BPN if that file is not there.
    if os.path.exists("hybrid_model.npz"):
        d = np.load("hybrid_model.npz")
        return d, BPN(w=d["w"]), True
    d = np.load("model.npz")
    return d, BPN(w=d["w"]), False

d, net, is_hybrid = load_model()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #7657D6;
    --primary-dark: #5C3CB5;
    --lavender: #F3EFFF;
    --mint: #DFF5EB;
    --bg: #F8F9FD;
    --text: #20233A;
    --muted: #85899E;
    color-scheme: light;
}

/* Font: scoped so icon fonts are not broken */
.stApp, .stApp p, .stApp label, .stApp input,
.stApp button,
.stApp span:not([class*="material"]):not([data-testid="stIconMaterial"]) {
    font-family: 'DM Sans', sans-serif;
}

/* Keep Streamlit's icon font (expander arrows etc.) from being overridden */
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}

/* Force light palette regardless of the user's Streamlit theme */
.stApp {
    background: var(--bg);
    color: var(--text);
}

.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1450px;
}

#MainMenu, footer {
    visibility: hidden;
}

hr {
    border-color: #E6E6F0 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #ECECF4;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.2rem;
}

section[data-testid="stSidebar"] h3 {
    color: #24243D !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 16px;
    font-weight: 700;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #51536A;
}

/* Brand */
.brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: #7657D6;
    letter-spacing: -1px;
}

.brand-sub {
    color: #999CAF;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
}

/* Hero */
.hero {
    background: linear-gradient(115deg, #7657D6 0%, #9579E9 60%, #B19AF3 100%);
    padding: 34px 38px;
    border-radius: 22px;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(118,87,214,0.16);
}

.stApp .hero h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF !important;
    margin: 0 0 10px 0;
    padding: 0;
    letter-spacing: -1px;
}

.stApp .hero p {
    font-size: 14px;
    color: #F0EAFE !important;
    margin: 0;
}

/* Section headings */
.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #24243D;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-sub {
    color: #9194A7;
    font-size: 13px;
    margin-bottom: 22px;
}

/* Input cards */
[data-testid="stNumberInput"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 12px 14px;
    border: 1px solid #ECECF4;
    box-shadow: 0 3px 12px rgba(35,35,70,0.025);
    margin-bottom: 10px;
}

[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] label p {
    color: #51536A !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="base-input"] {
    background: #F8F9FD !important;
    border-radius: 8px;
}

[data-testid="stNumberInput"] input {
    background: #F8F9FD !important;
    color: #282541 !important;
    -webkit-text-fill-color: #282541 !important;
    font-weight: 700 !important;
}

[data-testid="stNumberInput"] button {
    background: #F1EDFF !important;
    color: #7657D6 !important;
}

[data-testid="stNumberInput"] button:hover {
    background: #E4DCFF !important;
}

[data-testid="stNumberInput"] svg {
    fill: #7657D6;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(100deg, #7657D6, #9676E9);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(118,87,214,0.22);
    transition: all 0.2s ease;
}

.stButton > button p {
    color: white !important;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(100deg, #6242C0, #8061D8);
    transform: translateY(-2px);
    color: white !important;
    border: none;
}

.stButton > button:focus:not(:active) {
    color: white !important;
    border: none;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: #E7E2F8 !important;
}

[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #7657D6, #9676E9) !important;
}

/* st.info box */
[data-testid="stAlert"] {
    background: #F1EDFF !important;
    border: 1px solid #E0D8FA;
    border-radius: 14px;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] svg {
    color: #5C3CB5 !important;
    fill: #5C3CB5 !important;
}

/* Result cards */
.result-card {
    background: white;
    padding: 27px;
    border-radius: 18px;
    border: 1px solid #ECECF4;
    box-shadow: 0 5px 20px rgba(35,35,70,0.035);
    margin-top: 15px;
}

.result-label {
    color: #898CA0;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.3px;
}

.result-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 43px;
    font-weight: 800;
    color: #7657D6;
    margin: 10px 0;
}

.result-description {
    color: #818498;
    font-size: 13px;
}

.risk-high {
    background: #FFF0F0;
    color: #C34B57;
    padding: 13px 17px;
    border-radius: 12px;
    font-weight: 700;
}

.risk-low {
    background: #E8F7EF;
    color: #20865A;
    padding: 13px 17px;
    border-radius: 12px;
    font-weight: 700;
}

.info-card {
    background: #F1EDFF;
    padding: 18px;
    border-radius: 14px;
    color: #6552A2;
    font-size: 13px;
}

.footer {
    text-align: center;
    color: #A0A2B2;
    font-size: 11px;
    padding: 25px 0 5px;
}

/* Tabs (added for the multi-tab layout) */
button[data-baseweb="tab"] {
    color: #51536A;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #7657D6;
}

[data-baseweb="tab-highlight"] {
    background-color: #7657D6 !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #ECECF4;
    border-radius: 12px;
}

/* Mobile */
@media(max-width: 768px) {
    .block-container {
        padding: 1rem;
    }
    .hero {
        padding: 25px;
    }
    .stApp .hero h1 {
        font-size: 25px;
    }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">✦ GlycoSense</div>
    <div class="brand-sub">HEALTH INTELLIGENCE</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Explore")
    st.caption("Use the tabs on the main page: Risk Assessment, Overview, Data, "
               "Methodology, Results, Discussion and References.")

    st.markdown("---")

    st.markdown("### Model Information")

    st.markdown("""
    <div class="info-card">
        <b>GA + BPN + Fuzzy hybrid</b><br><br>
        A backpropagation neural network
        initialised by a Genetic Algorithm and
        combined with fuzzy clinical rules.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.caption("PROJECT · DIABETES PREDICTION")
    st.caption("Version 1.0")

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Your Health, Better Understood.</h1>
    <p>Explore your diabetes risk using an intelligent
    neural network powered by Genetic Algorithm optimization.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TABS
# --------------------------------------------------
TAB_NAMES = ["🩺 Risk Assessment", "📘 Overview", "🗂 Data", "🧠 Methodology",
             "📊 Results", "💡 Discussion", "📚 References"]
tab_assess, tab_overview, tab_data, tab_method, tab_results, tab_disc, tab_refs = st.tabs(TAB_NAMES)

with tab_assess:
    # --------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------
    st.markdown('<div class="section-title">Personal Health Assessment</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="section-sub">
    Enter your health indicators below to generate your risk assessment.
    </div>
    """, unsafe_allow_html=True)

    # Fields: int defaults/bounds -> integer input, float -> decimal input
    fields = [
        ("Pregnancies", 0, 17, 3, "Number of pregnancies"),
        ("Glucose", 40, 250, 120, "Plasma glucose concentration (mg/dL)"),
        ("Blood Pressure", 30, 130, 70, "Diastolic blood pressure (mm Hg)"),
        ("Skin Thickness", 5, 100, 25, "Triceps skin fold thickness (mm)"),
        ("Insulin", 10, 850, 80, "2-hour serum insulin (mu U/mL)"),
        ("BMI", 15.0, 70.0, 30.0, "Body mass index (kg/m²)"),
        ("Diabetes Pedigree Function", 0.05, 2.5, 0.4, "Genetic influence score"),
        ("Age", 21, 90, 33, "Age in years")
    ]

    vals = []

    # Two-column input layout
    left, right = st.columns(2, gap="large")

    for i, (name, lo, hi, dv, help_text) in enumerate(fields):
        col = left if i % 2 == 0 else right

        with col:
            is_int = isinstance(dv, int)
            value = st.number_input(
                name,
                min_value=lo,
                max_value=hi,
                value=dv,
                step=1 if is_int else 0.1,
                format="%d" if is_int else "%.2f",
                help=help_text,
                key=f"field_{i}"
            )
            vals.append(value)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    st.markdown("---")

    st.markdown("""
    <div class="section-title">Ready to analyze?</div>
    <div class="section-sub">
    Your inputs will be processed by the trained prediction model.
    </div>
    """, unsafe_allow_html=True)

    if st.button("✦  Generate Risk Assessment", width="stretch"):

        raw = np.array(vals, dtype=float)
        x = (raw - d["mean"]) / d["scale"]
        p = float(net.predict_proba(x.reshape(1, -1))[0, 0])
        fuzzy = None
        if is_hybrid:
            # fuzzy layer: clinical rules on the RAW values, fused with the network
            fsys, alpha = fz.decode(d["genes"])
            p_fz = float(fsys.score(raw.reshape(1, -1))[0])
            p = float((1 - alpha) * p + alpha * p_fz)
            level, _ = fz.risk_levels(np.array([p]))
            fuzzy = {"level": str(level[0]), "p_fz": p_fz, "alpha": float(alpha),
                     "rules": fsys.explain(raw)}

        st.session_state["prediction"] = p
        st.session_state["fuzzy"] = fuzzy

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    if "prediction" in st.session_state:

        p = st.session_state["prediction"]

        st.markdown("---")

        st.markdown("""
        <div class="section-title">Your Assessment Results</div>
        <div class="section-sub">
        Here's what the model's analysis indicates.
        </div>
        """, unsafe_allow_html=True)

        result_col, detail_col = st.columns([1, 1], gap="large")

        with result_col:

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Probability</div>
                <div class="result-value">{p:.1%}</div>
                <div class="result-description">
                    Model-estimated probability of diabetes
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(max(p, 0.0), 1.0))

        with detail_col:

            if p >= 0.5:
                st.markdown("""
                <div class="result-card">
                    <div class="result-label">Model Classification</div>
                    <br>
                    <div class="risk-high">
                        ● Higher predicted risk
                    </div>
                    <br>
                    <p style="color:#77798D;font-size:13px;">
                        The model's predicted probability is at or above
                        the 50% classification threshold.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="result-card">
                    <div class="result-label">Model Classification</div>
                    <br>
                    <div class="risk-low">
                        ● Lower predicted risk
                    </div>
                    <br>
                    <p style="color:#77798D;font-size:13px;">
                        The model's predicted probability is below
                        the 50% classification threshold. A low score does
                        not rule out diabetes: the model misses a sizeable
                        share of true cases (recall is roughly 0.6 at this
                        threshold).
                    </p>
                </div>
                """, unsafe_allow_html=True)

        fuzzy = st.session_state.get("fuzzy")
        if fuzzy:
            rules_html = "".join(
                f"<li>{txt} <span style='color:#9194A7'>(strength {s:.2f})</span></li>"
                for txt, s in fuzzy["rules"]) or "<li>No rule fired strongly.</li>"
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Fuzzy Risk Level</div>
                <div class="result-value">{fuzzy['level']}</div>
                <div class="result-description">
                    Score = {1 - fuzzy['alpha']:.0%} neural network +
                    {fuzzy['alpha']:.0%} fuzzy rules (rule-based score
                    {fuzzy['p_fz']:.0%}).
                </div>
                <br>
                <div class="result-label">Most active rules</div>
                <ul style="color:#77798D;font-size:13px;">{rules_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        st.info(
            "Important: This is an educational machine learning "
            "project, not a medical diagnosis or a clinically validated "
            "risk assessment. Consult a qualified healthcare professional "
            "for medical advice."
        )


# --------------------------------------------------
# SHARED HELPERS FOR THE INFORMATION TABS
# --------------------------------------------------
REPORT_FILE = "REPORT_SECTION_results_discussion.md"
REPORT_SECS = dict(split_sections(read_text(REPORT_FILE), "## "))


def head(title, sub=""):
    st.markdown(f'<div class="section-title">{title}</div>'
                f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def show_report(prefix):
    """Render one numbered section (e.g. '7.4') of the written report."""
    for k, v in REPORT_SECS.items():
        if k.startswith(prefix):
            st.markdown(f"#### {k}")
            st.markdown(v)
            return
    st.warning(f"Section {prefix} not found in {REPORT_FILE}.")


def md_expanders(path, expanded_first=1):
    """Render every '###' block of a generated results file in an expander."""
    secs = split_sections(read_text(path), "### ")
    if not secs:
        st.warning(f"{path} not found. Run the experiment script to create it.")
    for i, (h, b) in enumerate(secs):
        with st.expander(h, expanded=(i < expanded_first)):
            st.markdown(b)


def show_image(name, caption):
    """Figures live in figs/ (or next to the scripts, where they are first saved)."""
    for path in (os.path.join("figs", name), name):
        if os.path.exists(path):
            st.image(path, caption=caption, width="stretch")
            return
    st.info(f"{name} not found. Run the experiment script to create it.")


@st.cache_data
def load_df():
    try:
        return load_data()
    except Exception:
        return None


# --------------------------------------------------
# TAB: OVERVIEW
# --------------------------------------------------
with tab_overview:
    head("Project Overview",
         "A GA-optimised neural network with a fuzzy inference layer for diabetes risk.")
    st.markdown(OVERVIEW_MD)
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.markdown(OVERVIEW_TAIL_MD)

# --------------------------------------------------
# TAB: DATA
# --------------------------------------------------
with tab_data:
    head("Data and Preprocessing", "What the model learns from, and how it is prepared.")
    st.markdown(DATA_MD)
    df = load_df()
    if df is None:
        st.warning("diabetes.csv could not be loaded, so the live statistics are hidden.")
    else:
        st.markdown("### Features (live statistics from the loaded data)")
        tbl = pd.DataFrame(FEATURES, columns=["Feature", "Meaning", "Unit"])
        X_ = df.drop(columns="Outcome")
        tbl["Min"] = [round(float(X_.iloc[:, i].min()), 2) for i in range(8)]
        tbl["Mean"] = [round(float(X_.iloc[:, i].mean()), 2) for i in range(8)]
        tbl["Max"] = [round(float(X_.iloc[:, i].max()), 2) for i in range(8)]
        tbl["Zeros (treated as missing)"] = [
            str(int((X_.iloc[:, i] == 0).sum())) if X_.columns[i] in ZERO_COLS else "-"
            for i in range(8)]
        st.dataframe(tbl, hide_index=True, width="stretch")
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.markdown(f"**Rows:** {len(df)}")
            counts = df["Outcome"].map({0: "No diabetes", 1: "Diabetes"}).value_counts()
            st.bar_chart(counts)
        with c2:
            feat = st.selectbox("Compare a feature between the two classes",
                                list(X_.columns), index=1, key="explore_feat")
            vals_ = X_[feat].replace(0, np.nan) if feat in ZERO_COLS else X_[feat]
            fig, ax = plt.subplots(figsize=(6, 2.8))
            bins = np.histogram_bin_edges(vals_.dropna(), bins=25)
            for c_, lab, col in ((0, "No diabetes", "#7657D6"), (1, "Diabetes", "#E5726B")):
                ax.hist(vals_[df["Outcome"] == c_].dropna(), bins=bins, alpha=0.6,
                        label=lab, color=col)
            ax.set_xlabel(feat + (" (zeros removed)" if feat in ZERO_COLS else ""))
            ax.legend(frameon=False)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

# --------------------------------------------------
# TAB: METHODOLOGY
# --------------------------------------------------
with tab_method:
    head("Methodology", "How the neural network, genetic algorithm and fuzzy logic work.")
    m_nn, m_ga, m_fz, m_aug, m_eval = st.tabs(
        ["Neural network", "Genetic algorithm", "Fuzzy layer",
         "Augmentation & features", "Evaluation protocol"])

    with m_nn:
        st.markdown(NN_MD)

    with m_ga:
        st.markdown(GA_MD)
        st.graphviz_chart("""
        digraph GA { rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fillcolor="#FFF4D6", color="#B8860B",
                fontname="Helvetica", fontsize=11];
          edge [color="#85899E"];
          a [label="Random\\npopulation"]; b [label="Fitness of\\neach individual"];
          c [label="Tournament\\nselection"]; d [label="Uniform\\ncrossover"];
          e [label="Gaussian\\nmutation"]; f [label="Elitism +\\nnew generation"];
          a -> b -> c -> d -> e -> f; f -> b [label="repeat"]; }
        """, width="stretch")

    with m_fz:
        st.markdown(FUZZY_MD)
        st.markdown("#### Membership functions")
        fig, axs = plt.subplots(1, 4, figsize=(14, 2.8))
        rng_ = {"Glucose": (40, 220), "BMI": (15, 55), "Age": (20, 85), "DPF": (0, 1.5)}
        pal = ("#7657D6", "#E5A93B", "#E5726B")
        for ax, name in zip(axs, fz.INPUT_NAMES):
            xs = np.linspace(*rng_[name], 300)
            for (sn, p_), col in zip(fz.INPUTS[name][2].items(), pal):
                ax.plot(xs, fz.trap(xs, *p_), color=col, lw=2, label=sn)
            ax.set_title(name, fontsize=10)
            ax.set_ylim(-0.03, 1.05)
            ax.legend(frameon=False, fontsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### The 14 rules")
        st.dataframe(pd.DataFrame(
            {"Rule": [f"R{r + 1}" for r in range(fz.N_RULES)],
             "Meaning": [fz.rule_text(r) for r in range(fz.N_RULES)]}),
            hide_index=True, width="stretch")

        st.markdown("#### Try the fuzzy system on its own")
        st.caption("Uses only the untuned expert rules: no neural network involved.")
        s1, s2 = st.columns(2)
        with s1:
            g_ = st.slider("Glucose (mg/dL)", 40, 250, 135, key="fx_g")
            bmi_ = st.slider("BMI", 15.0, 60.0, 33.0, key="fx_bmi")
        with s2:
            age_ = st.slider("Age", 21, 90, 45, key="fx_age")
            dpf_ = st.slider("Diabetes pedigree function", 0.05, 2.5, 0.5, key="fx_dpf")
        xf = np.zeros((1, 8))
        xf[0, 1], xf[0, 5], xf[0, 6], xf[0, 7] = g_, bmi_, dpf_, age_
        esys = fz.FuzzySystem()
        score_ = float(esys.score(xf)[0])
        lvl_, mem_ = fz.risk_levels(np.array([score_]))
        mem_rows = [{"Input": n, "Set": sn, "Membership": round(float(v[0]), 2)}
                    for n, d_ in esys.memberships(xf).items() for sn, v in d_.items()]
        r1, r2 = st.columns([1, 1], gap="large")
        with r1:
            st.markdown(f"**Fuzzy risk score:** {score_:.0%} &nbsp;|&nbsp; "
                        f"**Level:** {lvl_[0]}")
            st.markdown("**Most active rules**")
            for txt, s_ in esys.explain(xf[0], top=5):
                st.markdown(f"- {txt} *(strength {s_:.2f})*")
        with r2:
            st.markdown("**Membership degrees**")
            st.dataframe(pd.DataFrame(mem_rows), hide_index=True,
                         width="stretch", height=250)

    with m_aug:
        st.markdown(AUG_MD)

    with m_eval:
        show_report("7.1")
        show_report("7.2")

# --------------------------------------------------
# TAB: RESULTS
# --------------------------------------------------
with tab_results:
    head("Results", "Every number here is read from the files the experiments generated.")
    r_acc, r_hyb, r_syn = st.tabs(
        ["Accuracy study", "Hybrid (fuzzy) study", "Synthetic data & GA selection"])

    with r_acc:
        show_report("7.3")
        show_image("fig_accuracy_models.png",
                   "Accuracy per model: filled = threshold tuned on validation, hollow = 0.5")
        st.markdown("#### Full generated tables (results_accuracy.md)")
        md_expanders("results_accuracy.md", expanded_first=0)

    with r_hyb:
        show_report("7.6")
        c1, c2 = st.columns(2)
        with c1:
            show_image("fig_hybrid_comparison.png", "Test-set comparison of the configurations")
            show_image("fig_hybrid_ablation.png", "Ablation: network start x fuzzy mode")
        with c2:
            show_image("fig_hybrid_confusion.png", "Confusion matrices, pooled over seeds")
            show_image("fig_risk_levels.png", "Observed diabetes rate per fuzzy risk level")
        show_image("fig_fuzzy_system.png",
                   "Expert membership functions, GA-tuned rule weights and GA convergence")
        st.markdown("#### Full generated tables (results_hybrid.md)")
        md_expanders("results_hybrid.md", expanded_first=0)

    with r_syn:
        st.markdown("#### How realistic is the synthetic data?")
        show_image("fig_synthetic_realism.png",
                   "Correlation of real vs synthetic rows, and per-feature KS distance")
        st.markdown("Gaussian rows keep the class ratio and most correlations; Insulin is the "
                    "weak spot because the real column has a spike of median-imputed values.")
        st.markdown("#### What the feature-selecting GA kept")
        show_image("fig_ga_feature_selection.png",
                   "Fraction of outer folds in which each feature was selected")

# --------------------------------------------------
# TAB: DISCUSSION
# --------------------------------------------------
with tab_disc:
    head("Discussion", "What the results mean, honestly.")
    show_report("7.4")
    show_report("7.5")
    show_report("7.7")
    show_report("7.8")
    st.markdown("#### Future work")
    st.markdown("""
* Larger or additional datasets and **external validation**; 768 rows limit what any model can show.
* **Nested cross-validation** for the GA, and a matched-compute baseline (same number of
  fitness evaluations for the plain network).
* Probability **calibration** and a cost-sensitive threshold chosen with clinicians.
* Restricting the fuzzy GA (fewer genes) or tuning it by cross-validation to reduce overfitting.
""")

# --------------------------------------------------
# TAB: REFERENCES
# --------------------------------------------------
with tab_refs:
    head("References", "Dataset, methods and statistical tests used in this project.")
    st.markdown(REFERENCES_MD)
    st.caption("Source files: diabetes_ga_bpn.py (network and GA), fuzzy_layer.py, augment.py, "
               "features.py, hybrid_experiment.py, accuracy_experiment.py.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    GLYCOSENSE · BUILT WITH MACHINE LEARNING
    <br><br>
    For educational and research purposes only.
</div>
""", unsafe_allow_html=True)