"""
NHANES ALT Risk Assistant — Streamlit UI + CLI entrypoint.

Streamlit:  streamlit run src/07_chat.py
CLI:        python src/07_chat.py --cli
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── CLI mode ──────────────────────────────────────────────────────────────────

def run_cli() -> None:
    from agents import Orchestrator

    print("\n=== NHANES ALT Risk Assistant (CLI) ===")
    print("Type 'quit' to exit.\n")

    try:
        orchestrator = Orchestrator()
    except EnvironmentError as e:
        print(f"Error: {e}")
        sys.exit(1)

    history: list[dict] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("Goodbye.")
            break
        response, agent = orchestrator.chat(user_input, history)
        print(f"\n[{agent}]\n{response}\n")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        if len(history) > 20:
            history = history[-20:]


# ── Streamlit mode ────────────────────────────────────────────────────────────

def run_streamlit() -> None:
    import streamlit as st

    FIGURES_DIR   = PROJECT_ROOT / "figures"
    REPORTS_P2_DIR = PROJECT_ROOT / "reports" / "phase2_analysis"
    REPORTS_P3_DIR = PROJECT_ROOT / "reports" / "phase3_covid"

    # Phase 1 & 2 figures — filenames include subfolder prefix
    FIGURES_PHASE12 = [
        ("Forest Plot — Odds Ratios (Updated)",  "phase2_analysis/fig_forest_plot_OR.png"),
        ("Extended Hypothesis Forest Plot",      "phase2_analysis/fig_extended_forest_plot.png"),
        ("Waist Circumference by ALT Status",    "phase2_analysis/fig_waist_by_alt.png"),
        ("Triglycerides by ALT Status",          "phase2_analysis/fig_triglycerides_by_alt.png"),
        ("SHAP Feature Importance",              "phase2_analysis/fig_shap_importance.png"),
        ("SHAP Summary (Beeswarm)",              "phase2_analysis/fig_shap_summary.png"),
        ("SHAP Waterfall — High Risk",           "phase2_analysis/fig_shap_waterfall_high.png"),
        ("SHAP Waterfall — Low Risk",            "phase2_analysis/fig_shap_waterfall_low.png"),
        ("ROC & Calibration Curves",             "phase2_analysis/fig_roc_calibration.png"),
        ("ALT Distribution",                     "phase1_eda/fig_alt_distribution.png"),
        ("BMI by ALT Status",                    "phase1_eda/fig_bmi_by_alt.png"),
        ("Prevalence by Race/Ethnicity",         "phase1_eda/fig_prevalence_by_race.png"),
        ("Blood Metals Distribution",            "phase1_eda/fig_metals_distribution.png"),
        ("Correlation Heatmap",                  "phase1_eda/fig_correlation_heatmap.png"),
    ]

    # Phase 3 figures (Pre vs Post COVID comparison)
    FIGURES_PHASE3 = [
        ("Full Dashboard (4-panel)",             "phase3_covid/fig_phase3_dashboard.png"),
        ("COVID Impact Summary",                 "phase3_covid/fig_covid_impact_summary.png"),
        ("Prevalence Shifts — All Variables",    "phase3_covid/fig_prevalence_comparison.png"),
        ("Risk Factor Shifts",                   "phase3_covid/fig_riskfactor_comparison.png"),
        ("Model OR Comparison Forest Plot",      "phase3_covid/fig_model_comparison_forest.png"),
        ("Wellness Variables — OR Comparison",   "phase3_covid/fig_wellness_forest_plot.png"),
        ("SHAP Importance Comparison",           "phase3_covid/fig_shap_comparison.png"),
        ("ROC Curves — Both Cohorts",            "phase3_covid/fig_roc_comparison.png"),
    ]

    SAMPLE_QUESTIONS = [
        "What is the OR for waist circumference and triglycerides?",
        "50-year-old diabetic male, waist 102 cm, triglycerides 220 — what's his risk?",
        "Run the analysis for diabetic women",
        "Which factors matter most for ALT elevation?",
        "What's the prevalence in Hispanic males over 40?",
        "How does the ML model perform?",
        "What does the safety report say about heavy metals?",
        "Compare risk: healthy 30F waist 78cm vs diabetic 55M waist 105cm triglycerides 250",
        "What changed in liver risk between 2017-2018 and 2021-2023?",
        "Why did male sex become more significant post-COVID?",
        "Did depression or sedentary behavior predict elevated ALT post-COVID?",
    ]

    REPORT_EXAMPLES = [
        "52-year-old diabetic male, waist 104 cm, triglycerides 210, ever smoker",
        "35-year-old female, waist 82 cm, no diabetes, triglycerides 95",
        "Metabolic syndrome and liver injury risk across the U.S. population",
        "Impact of waist circumference and triglycerides on ALT elevation risk",
        "Non-Hispanic Black adults over 50",
    ]

    st.set_page_config(
        page_title="NHANES ALT Risk Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap');

    /* Base: force dark text on the entire app */
    html, body {
        color: #1A2230 !important;
        background: #F5F7FA !important;
    }
    .stApp {
        background: #F5F7FA !important;
        color: #1A2230 !important;
    }

    /* All markdown content: dark text */
    .stMarkdown, .stMarkdown *,
    .stText, .stText *,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * {
        color: #1A2230 !important;
    }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #0D1B2A !important;
    }

    /* Caption */
    .stCaption, .stCaption p { color: #5A6675 !important; }

    /* Font — only text elements */
    p, li, label, input, textarea, button {
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide default chrome */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; }

    /* ══════════════ SIDEBAR ══════════════ */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #E6EAF0 !important;
        min-width: 300px !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

    .brand-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1B2A3D 100%);
        padding: 1.4rem 1.4rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .brand-title { font-size: 1.2rem; font-weight: 700; color: #ffffff; }
    .brand-sub {
        font-size: 0.67rem; font-weight: 600; color: #9FB2C9;
        text-transform: uppercase; letter-spacing: 0.09em; margin-top: 0.2rem;
    }

    .sec-label {
        display: block;
        font-size: 0.67rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #5A6675;
        margin: 1rem 0 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff; border: 1px solid #E6EAF0;
        border-radius: 12px; padding: 0.75rem 0.85rem;
        box-shadow: 0 1px 3px rgba(13,27,42,0.05);
        margin-bottom: 0.5rem;
    }
    .metric-lbl {
        font-size: 0.64rem; font-weight: 700; color: #5A6675;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.45rem; font-weight: 700; color: #0D1B2A;
        margin-top: 0.1rem; letter-spacing: -0.01em; line-height: 1.2;
    }

    /* Pill buttons — sidebar only */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        background: #F3F6FB !important;
        color: #1A2230 !important;
        border: 1px solid #E6EAF0 !important;
        border-radius: 999px !important;
        padding: 0.48rem 1rem !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
        margin-bottom: 0.25rem !important;
        white-space: normal !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #0D1B2A !important;
        color: #ffffff !important;
        border-color: #0D1B2A !important;
    }

    /* Primary button (Generate Report) */
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: #0D1B2A !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.4rem !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        width: auto !important;
        cursor: pointer !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background: #1B2A3D !important;
    }

    /* Regular main-area buttons (example chips, Clear) */
    .stMain .stButton > button:not([kind="primary"]) {
        background: #F3F6FB !important;
        color: #1A2230 !important;
        border: 1px solid #E6EAF0 !important;
        border-radius: 999px !important;
        padding: 0.45rem 1rem !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        white-space: normal !important;
        transition: all 0.15s ease !important;
        width: auto !important;
    }
    .stMain .stButton > button:not([kind="primary"]):hover {
        background: #0D1B2A !important;
        color: #ffffff !important;
        border-color: #0D1B2A !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 10px !important;
        border: 1px solid #E6EAF0 !important;
        background: #ffffff !important;
        color: #1A2230 !important;
    }

    /* Expander — fix arrow overlapping title */
    [data-testid="stExpander"] {
        border: 1px solid #E6EAF0 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
        margin-bottom: 0.4rem !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] details summary {
        padding: 0.65rem 0.95rem !important;
        background: #ffffff !important;
        cursor: pointer !important;
    }
    [data-testid="stExpander"] details summary > span {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #1A2230 !important;
    }
    [data-testid="stExpander"] details summary svg {
        color: #5A6675 !important;
    }
    [data-testid="stExpander"] details > div {
        background: #ffffff !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stExpander"] details > div p,
    [data-testid="stExpander"] details > div li,
    [data-testid="stExpander"] details > div span,
    [data-testid="stExpander"] details > div td,
    [data-testid="stExpander"] details > div th {
        color: #1A2230 !important;
        font-size: 0.86rem !important;
        line-height: 1.65 !important;
    }
    [data-testid="stExpander"] details > div h1,
    [data-testid="stExpander"] details > div h2,
    [data-testid="stExpander"] details > div h3 {
        color: #0D1B2A !important;
    }

    /* ══════════════ MAIN AREA ══════════════ */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0D1B2A !important;
        letter-spacing: -0.02em !important;
    }

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        border-bottom: 1px solid #E6EAF0 !important;
        padding-bottom: 0 !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.55rem 1.1rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #5A6675 !important;
        border: none !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] p,
    [data-testid="stTabs"] [data-baseweb="tab"] span {
        color: #5A6675 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #ffffff !important;
        color: #0D1B2A !important;
        border-top: 2px solid #0D1B2A !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] p,
    [data-testid="stTabs"] [aria-selected="true"] span {
        color: #0D1B2A !important;
    }
    [data-testid="stTabContent"] {
        background: #ffffff !important;
        border: 1px solid #E6EAF0 !important;
        border-top: none !important;
        border-radius: 0 0 14px 14px !important;
        padding: 1.5rem !important;
        color: #1A2230 !important;
    }
    [data-testid="stTabContent"] * {
        color: #1A2230 !important;
    }
    [data-testid="stTabContent"] h1,
    [data-testid="stTabContent"] h2,
    [data-testid="stTabContent"] h3,
    [data-testid="stTabContent"] h4,
    [data-testid="stTabContent"] strong,
    [data-testid="stTabContent"] b {
        color: #0D1B2A !important;
    }
    [data-testid="stTabContent"] p,
    [data-testid="stTabContent"] li,
    [data-testid="stTabContent"] span,
    [data-testid="stTabContent"] label,
    [data-testid="stTabContent"] div {
        color: #1A2230 !important;
    }
    [data-testid="stTabContent"] textarea {
        color: #1A2230 !important;
        background: #ffffff !important;
    }
    [data-testid="stTabContent"] textarea::placeholder {
        color: #9AA5B4 !important;
    }

    /* Route badge */
    .route-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.3rem 0.85rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .route-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .badge-qa  { background: rgba(33,150,243,0.12); color: #1565C0; }
    .badge-qa .route-dot  { background: #2196F3; }
    .badge-an  { background: rgba(123,47,190,0.12); color: #6A1B9A; }
    .badge-an .route-dot  { background: #7B2FBE; }

    /* Chat messages — force ALL content dark and readable */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #E6EAF0 !important;
        border-radius: 16px !important;
        padding: 0.9rem 1.1rem !important;
        box-shadow: 0 1px 3px rgba(13,27,42,0.04) !important;
        margin-bottom: 0.5rem !important;
    }
    /* Nuclear option: every element inside a chat bubble gets dark text */
    [data-testid="stChatMessage"] * {
        color: #1A2230 !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] div {
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
    }
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #0D1B2A !important;
        font-weight: 700 !important;
    }
    /* Tables inside chat */
    [data-testid="stChatMessage"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 0.6rem 0 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stChatMessage"] th {
        background: #F0F4F8 !important;
        color: #0D1B2A !important;
        font-weight: 700 !important;
        padding: 0.4rem 0.65rem !important;
        border: 1px solid #D1D9E6 !important;
        text-align: left !important;
    }
    [data-testid="stChatMessage"] td {
        color: #1A2230 !important;
        padding: 0.35rem 0.65rem !important;
        border: 1px solid #E6EAF0 !important;
    }
    [data-testid="stChatMessage"] tr:nth-child(even) td {
        background: #F8FAFC !important;
    }
    [data-testid="stChatMessage"] code {
        background: #F0F4F8 !important;
        color: #0D1B2A !important;
        padding: 0.1rem 0.35rem !important;
        border-radius: 4px !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stChatMessage"] blockquote {
        border-left: 3px solid #2196F3 !important;
        padding-left: 0.75rem !important;
        color: #3A4A5C !important;
        margin: 0.5rem 0 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] > div {
        background: #ffffff !important;
        border: 1px solid #D1D9E6 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px rgba(13,27,42,0.07) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #1A2230 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #9AA5B4 !important; }

    /* Text area (report tab) */
    [data-testid="stTextArea"] textarea {
        border-radius: 12px !important;
        border: 1px solid #D1D9E6 !important;
        font-size: 0.93rem !important;
        color: #1A2230 !important;
        background: #ffffff !important;
        line-height: 1.55 !important;
        caret-color: #0D1B2A !important;
    }
    [data-testid="stTextArea"] textarea::placeholder {
        color: #9AA5B4 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stTextArea"] label {
        color: #1A2230 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Report output */
    .report-output {
        background: #ffffff;
        border: 1px solid #E6EAF0;
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        line-height: 1.7;
        color: #1A2230;
        margin-top: 1rem;
    }
    .report-output h1 { font-size: 1.4rem !important; color: #0D1B2A !important; margin-bottom: 0.3rem !important; }
    .report-output h2 { font-size: 1.1rem !important; color: #0D1B2A !important; margin-top: 1.2rem !important; border-bottom: 1px solid #E6EAF0; padding-bottom: 0.3rem; }
    .report-output h3 { font-size: 0.97rem !important; color: #1A2230 !important; }
    .report-output p, .report-output li { color: #1A2230 !important; font-size: 0.93rem !important; }
    .report-output strong { color: #0D1B2A !important; }

    /* Custom report toggles (replaces st.expander) */
    .report-toggle {
        display: none; /* hidden — only the button is interactive */
    }
    .report-content {
        background: #ffffff;
        border: 1px solid #E6EAF0;
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 0.85rem 1rem;
        font-size: 0.85rem;
        color: #1A2230;
        line-height: 1.65;
        margin-bottom: 0.4rem;
        margin-top: -0.3rem;
    }
    .report-content h1, .report-content h2, .report-content h3 {
        color: #0D1B2A; margin-top: 0.8rem;
    }
    .report-content p { color: #1A2230; margin-bottom: 0.4rem; }
    .report-content li { color: #1A2230; }
    .report-content table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .report-content th { background: #F5F7FA; color: #0D1B2A; padding: 0.35rem 0.5rem; border: 1px solid #E6EAF0; }
    .report-content td { padding: 0.3rem 0.5rem; border: 1px solid #E6EAF0; color: #1A2230; }

    /* Override the pill button style ONLY for report toggle buttons */
    [data-testid="stSidebar"] .report-btn > button {
        border-radius: 10px !important;
        background: #ffffff !important;
        border: 1px solid #E6EAF0 !important;
        color: #1A2230 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.65rem 1rem !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .report-btn > button:hover {
        background: #F5F7FA !important;
        color: #0D1B2A !important;
        border-color: #D1D9E6 !important;
        transform: none !important;
    }

    /* Sidebar footer */
    .side-foot {
        margin-top: 1.2rem; padding-top: 0.8rem;
        border-top: 1px solid #E6EAF0;
        font-size: 0.72rem; color: #8A96A3; text-align: center;
    }

    /* Warning banner */
    .warn-banner {
        background: #FFF8E1; border: 1px solid #FFD54F;
        border-radius: 10px; padding: 0.7rem 1rem;
        font-size: 0.88rem; color: #5D4037; margin-bottom: 1rem;
    }

    /* New discoveries callout */
    .discovery-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
        border: 1.5px solid #66BB6A;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
    }
    .discovery-title {
        font-size: 0.68rem; font-weight: 700; color: #2E7D32;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;
    }
    .discovery-item {
        font-size: 0.82rem; color: #1B5E20; line-height: 1.5; margin: 0.15rem 0;
    }
    .discovery-item strong { color: #1B5E20; }

    /* Hypothesis summary table */
    .hypo-table {
        width: 100%; font-size: 0.78rem; border-collapse: collapse;
        margin-top: 0.5rem;
    }
    .hypo-table th {
        background: #F0F4F8; color: #0D1B2A; font-weight: 700;
        padding: 0.35rem 0.5rem; border: 1px solid #D1D9E6; text-align: left;
    }
    .hypo-table td {
        color: #1A2230; padding: 0.3rem 0.5rem;
        border: 1px solid #E6EAF0;
    }
    .hypo-table tr:nth-child(even) td { background: #F8FAFC; }
    .sig-yes { color: #2E7D32; font-weight: 700; }
    .sig-no  { color: #9AA5B4; }

    /* ── Phase 3 comparison styles ────────────────────── */
    .cohort-header {
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    }
    .cohort-header-17 {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border: 1.5px solid #90CAF9;
    }
    .cohort-header-21 {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border: 1.5px solid #FFCC80;
    }
    .cohort-title {
        font-size: 1rem; font-weight: 700; color: #0D1B2A; margin-bottom: 0.5rem;
    }
    .cohort-stat { font-size: 0.83rem; color: #1A2230; line-height: 1.7; }
    .cohort-stat strong { color: #0D1B2A; }

    .change-up   { color: #B22222; font-weight: 700; }
    .change-down { color: #1B5E20; font-weight: 700; }
    .change-neu  { color: #5A6675; }

    .compare-card {
        background: #ffffff; border: 1px solid #E6EAF0;
        border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    }
    .compare-card h4 { color: #0D1B2A; font-size: 1rem; margin: 0 0 0.6rem; }

    .phase-section-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #5A6675; margin: 0.6rem 0 0.3rem;
        display: block; border-top: 1px solid #E6EAF0; padding-top: 0.6rem;
    }
    .fig-phase-header {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 0.3rem 0.6rem; border-radius: 6px;
        display: inline-block; margin-bottom: 0.35rem;
    }
    .fig-phase12 { background: #E3F2FD; color: #1565C0; }
    .fig-phase3  { background: #FFF3E0; color: #B25A00; }
    </style>
    """, unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────────
    for key, default in [
        ("messages",          []),
        ("history",           []),
        ("last_agent",        None),
        ("orchestrator",      None),
        ("report_generator",  None),
        ("generated_report",  None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Init agents ────────────────────────────────────────────────────────────
    api_key     = os.environ.get("ANTHROPIC_API_KEY")
    api_missing = not bool(api_key)

    if not api_missing:
        if st.session_state.orchestrator is None:
            try:
                from agents import Orchestrator
                st.session_state.orchestrator = Orchestrator()
            except Exception:
                api_missing = True
        if st.session_state.report_generator is None:
            try:
                from agents import ReportGenerator
                st.session_state.report_generator = ReportGenerator()
            except Exception as e:
                st.session_state._report_init_error = str(e)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:

        st.markdown("""
        <div class="brand-header">
            <div class="brand-title">🔬 NHANES Liver Risk</div>
            <div class="brand-sub">Kenvue Portfolio Project</div>
        </div>
        """, unsafe_allow_html=True)

        # Study stats — both cohorts
        st.markdown('<span class="sec-label">Study at a Glance</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-bottom:0.5rem">
          <div class="metric-card" style="border-top:3px solid #4C72B0">
            <div class="metric-lbl" style="color:#1565C0">2017-2018 (Pre-COVID)</div>
            <div class="metric-val">3,543</div>
            <div style="font-size:0.72rem;color:#5A6675;margin-top:0.1rem">ALT>40: 8.15%</div>
          </div>
          <div class="metric-card" style="border-top:3px solid #DD8452">
            <div class="metric-lbl" style="color:#B25A00">2021-2023 (Post-COVID)</div>
            <div class="metric-val">3,934</div>
            <div style="font-size:0.72rem;color:#5A6675;margin-top:0.1rem">ALT>40: 7.36%</div>
          </div>
          <div class="metric-card">
            <div class="metric-lbl">ML AUC (2017)</div>
            <div class="metric-val">97.3%</div>
          </div>
          <div class="metric-card">
            <div class="metric-lbl">ML AUC (2021)</div>
            <div class="metric-val">95.5%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # New Discoveries callout
        st.markdown('<span class="sec-label">New Discoveries</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="discovery-card">
            <div class="discovery-title">✦ Extended Hypothesis Findings</div>
            <div class="discovery-item">
                <strong>Waist circumference</strong> — OR 1.027/cm &nbsp;(p = 0.012 ★)<br>
                Visceral fat drives hepatic steatosis (NAFLD pathway)
            </div>
            <div class="discovery-item" style="margin-top:0.4rem">
                <strong>Triglycerides</strong> — OR 1.77/log-unit &nbsp;(p = 0.018 ★)<br>
                200 mg/dL → ~27% higher odds vs. 100 mg/dL
            </div>
            <div class="discovery-item" style="margin-top:0.4rem; color:#5A6675">
                ✗ Alcohol, education, heavy metals — not significant
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Phase 3 COVID comparison callout
        st.markdown('<span class="sec-label">Phase 3: Post-COVID Findings</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="discovery-card" style="border-left: 4px solid #DD8452;">
            <div class="discovery-title" style="color:#B25A00;">⚡ Pre vs Post COVID Shifts</div>
            <div class="discovery-item">
                <strong>Male sex</strong> became highly significant post-COVID<br>
                OR: 2.03 → <strong>2.93</strong> &nbsp;(p: 0.065 → <strong>0.0006</strong> ★)
            </div>
            <div class="discovery-item" style="margin-top:0.4rem">
                <strong>Triglycerides effect strengthened</strong><br>
                OR: 1.77 → <strong>1.97</strong> &nbsp;(p = 0.014 ★)
            </div>
            <div class="discovery-item" style="margin-top:0.4rem; color:#5A6675">
                Depression +3.6pp &nbsp;|&nbsp; Sedentary +3.4pp — but NOT liver predictors<br>
                ALT>40 prevalence: 8.15% → 7.36% (slight improvement)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Figure viewer — split by phase
        st.markdown('<span class="sec-label">Figure Viewer</span>', unsafe_allow_html=True)
        fig_phase = st.radio(
            "Phase",
            ["Phase 1 & 2 — 2017-2018", "Phase 3 — Pre vs Post COVID"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if "Phase 3" in fig_phase:
            st.markdown('<span class="fig-phase-header fig-phase3">🦠 Phase 3 — Pre/Post COVID</span>', unsafe_allow_html=True)
            avail3 = [(lbl, fn) for lbl, fn in FIGURES_PHASE3 if (FIGURES_DIR / fn).exists()]
            if avail3:
                lbl3 = [l for l, _ in avail3]
                fn3  = [f for _, f in avail3]
                ch3  = st.selectbox("fig3", lbl3, label_visibility="collapsed")
                st.image(str(FIGURES_DIR / fn3[lbl3.index(ch3)]), use_container_width=True)
            else:
                st.caption("Phase 3 figures not found. Run scripts 13-17 first.")
        else:
            st.markdown('<span class="fig-phase-header fig-phase12">📊 Phase 1 & 2 — 2017-2018</span>', unsafe_allow_html=True)
            avail12 = [(lbl, fn) for lbl, fn in FIGURES_PHASE12 if (FIGURES_DIR / fn).exists()]
            if avail12:
                lbl12 = [l for l, _ in avail12]
                fn12  = [f for _, f in avail12]
                ch12  = st.selectbox("fig12", lbl12, label_visibility="collapsed")
                st.image(str(FIGURES_DIR / fn12[lbl12.index(ch12)]), use_container_width=True)
            else:
                st.caption("Phase 1 & 2 figures not found. Run the pipeline first.")

        # Reports — custom toggle to avoid Streamlit expander icon bug
        st.markdown('<span class="sec-label">Project Overall Summary Reports</span>', unsafe_allow_html=True)

        report_defs = [
            ("show_technical",   "📄  Technical Report",        "technical.md"),
            ("show_safety",      "🛡️  Safety Summary",          "safety_summary.md"),
            ("show_methods",     "⚖️  Methods & Limitations",   "methods_limitations.md"),
        ]
        for state_key, title, fname in report_defs:
            if state_key not in st.session_state:
                st.session_state[state_key] = False
            path = REPORTS_P2_DIR / fname
            if not path.exists():
                continue
            is_open = st.session_state[state_key]
            arrow   = "▼" if is_open else "▶"
            st.markdown('<div class="report-btn">', unsafe_allow_html=True)
            if st.button(f"{arrow}  {title}", key=f"btn_{state_key}"):
                st.session_state[state_key] = not is_open
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            if is_open:
                content = path.read_text(encoding="utf-8")
                st.markdown(f'<div class="report-content">{content}</div>', unsafe_allow_html=True)

        # Hypothesis summary table
        hypo_path = REPORTS_P2_DIR / "hypothesis_test_all_variables.csv"
        if hypo_path.exists():
            if "show_hypothesis" not in st.session_state:
                st.session_state.show_hypothesis = False
            is_open = st.session_state.show_hypothesis
            arrow   = "▼" if is_open else "▶"
            st.markdown('<div class="report-btn">', unsafe_allow_html=True)
            if st.button(f"{arrow}  📋  Hypothesis Test Summary", key="btn_hypothesis"):
                st.session_state.show_hypothesis = not is_open
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            if is_open:
                import pandas as _pd
                hypo_df = _pd.read_csv(hypo_path)
                display_cols = ["label", "category",
                                "orig_OR", "orig_p", "orig_stars",
                                "updated_OR", "updated_p", "updated_stars",
                                "extended_OR", "extended_p", "extended_stars",
                                "significant_in_any_model"]
                hypo_df = hypo_df[display_cols].rename(columns={
                    "label": "Variable", "category": "Category",
                    "orig_OR": "Orig OR", "orig_p": "Orig p", "orig_stars": "",
                    "updated_OR": "Upd OR", "updated_p": "Upd p", "updated_stars": " ",
                    "extended_OR": "Ext OR", "extended_p": "Ext p", "extended_stars": "  ",
                    "significant_in_any_model": "Significant?",
                })
                st.markdown(
                    "<small style='color:#5A6675'>Orig = 9-predictor (BMI) · Upd = 10-predictor "
                    "(waist+trig) · Ext = extended hypothesis test</small>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    hypo_df,
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                )

        # Sample questions (for chat tab)
        st.markdown('<span class="sec-label">Quick Questions</span>', unsafe_allow_html=True)
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sq_{hash(q)}"):
                st.session_state._prefill = q

        st.markdown('<div class="side-foot">Powered by Claude AI &nbsp;+&nbsp; NHANES data</div>', unsafe_allow_html=True)

    # ── MAIN ──────────────────────────────────────────────────────────────────
    st.title("NHANES ALT Risk Assistant")
    st.markdown(
        '<p style="color:#5A6675;font-size:0.96rem;margin-top:-0.4rem;margin-bottom:0.4rem;line-height:1.5;">'
        "Ask about liver-injury risk factors, get personalized risk estimates, or explore the "
        "pre/post COVID comparison. Covers NHANES 2017-2018 and 2021-2023."
        "</p>"
        '<div style="display:flex;gap:0.6rem;margin-bottom:1rem;flex-wrap:wrap;">'
        '<p style="font-size:0.81rem;color:#1565C0;background:#E3F2FD;border:1px solid #90CAF9;'
        'border-radius:8px;padding:0.4rem 0.85rem;margin:0;line-height:1.5;flex:1;min-width:200px">'
        "📘 <strong>2017-2018:</strong> Waist (OR 1.03 ★) + Triglycerides (OR 1.77 ★) — dominant metabolic pathway"
        "</p>"
        '<p style="font-size:0.81rem;color:#B25A00;background:#FFF3E0;border:1px solid #FFCC80;'
        'border-radius:8px;padding:0.4rem 0.85rem;margin:0;line-height:1.5;flex:1;min-width:200px">'
        "📙 <strong>2021-2023:</strong> Male sex now OR 2.93 ★★★ — strongest new post-COVID signal"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if api_missing:
        st.markdown(
            '<div class="warn-banner">⚠️ <strong>ANTHROPIC_API_KEY not set.</strong> '
            'Run: <code>export ANTHROPIC_API_KEY=sk-ant-...</code> then restart.</div>',
            unsafe_allow_html=True,
        )

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_chat, tab_covid, tab_report = st.tabs([
        "💬  Chat",
        "🦠  Phase 3: COVID Comparison",
        "📊  Generate Report",
    ])

    # ── TAB 1: CHAT ───────────────────────────────────────────────────────────
    with tab_chat:

        # Route badge
        if st.session_state.last_agent:
            agent_name = st.session_state.last_agent
            cls = "badge-qa" if "Q&A" in agent_name else "badge-an"
            st.markdown(
                f'<div class="route-badge {cls}">'
                f'<span class="route-dot"></span>Routed to: {agent_name}</div>',
                unsafe_allow_html=True,
            )

        # Welcome message
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    "Hi! I can answer questions about the NHANES 2017–2018 liver study, or estimate "
                    "an individual's risk of elevated ALT. Try a quick question from the sidebar, "
                    "or describe a health profile — for example: "
                    "*'45-year-old diabetic male, waist 100 cm, triglycerides 200 mg/dL — what is his risk?'*"
                )

        # Chat history
        for msg in st.session_state.messages:
            avatar = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        # Sidebar pill prefill
        prefill = st.session_state.pop("_prefill", None)

        user_input = st.chat_input(
            "Ask about liver health risk factors...",
            disabled=api_missing,
        ) or prefill

        if user_input:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        response, agent = st.session_state.orchestrator.chat(
                            user_input, st.session_state.history,
                        )
                        st.session_state.last_agent = agent
                    except Exception as e:
                        response = f"Something went wrong: {e}"
                        agent    = "Error"
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.history.append({"role": "user",      "content": user_input})
            st.session_state.history.append({"role": "assistant", "content": response})
            if len(st.session_state.history) > 12:
                st.session_state.history = st.session_state.history[-12:]
            st.rerun()

    # ── TAB 2: COVID COMPARISON ───────────────────────────────────────────────
    with tab_covid:
        import pandas as _pd

        st.markdown("### Phase 3: Pre vs Post COVID — NHANES 2017-2018 vs 2021-2023")
        st.markdown(
            '<p style="color:#5A6675;font-size:0.9rem;margin-top:-0.4rem;margin-bottom:1rem;">'
            "Same 10-predictor survey-weighted logistic model applied to both cohorts. "
            "Same exclusion criteria, same feature engineering. "
            "Differences in results reflect genuine population-level changes."
            "</p>",
            unsafe_allow_html=True,
        )

        # ── Side-by-side cohort header cards ─────────────────────────────────
        col_17, col_21 = st.columns(2)
        with col_17:
            st.markdown("""
            <div class="cohort-header cohort-header-17">
              <div class="cohort-title">📘 2017-2018 &nbsp;(Pre-COVID baseline)</div>
              <div class="cohort-stat">
                <strong>N = 3,543</strong> adults  |  <strong>ALT &gt; 40:</strong> 8.15% weighted<br>
                Significant predictors: <strong>Age, Waist, Triglycerides</strong><br>
                Male sex: OR 2.03, p = 0.065 (borderline NS)<br>
                Diabetes: OR 1.98, p = 0.070 (borderline NS)<br>
                ML AUC-ROC: <strong>0.970</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with col_21:
            st.markdown("""
            <div class="cohort-header cohort-header-21">
              <div class="cohort-title">📙 2021-2023 &nbsp;(Post-COVID)</div>
              <div class="cohort-stat">
                <strong>N = 3,934</strong> adults  |  <strong>ALT &gt; 40:</strong> 7.36% weighted<br>
                Significant predictors: <strong>Age, Waist, Triglycerides, Male sex ★</strong><br>
                Male sex: OR <strong>2.93</strong>, p = <strong>0.0006</strong> (newly significant)<br>
                Diabetes: OR 0.90, p = 0.74 (no longer significant)<br>
                ML AUC-ROC: <strong>0.955</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Sub-sections ──────────────────────────────────────────────────────
        sub_overview, sub_model, sub_wellness, sub_ml = st.tabs([
            "📊 Overview",
            "📈 Model Comparison",
            "🧠 Wellness Variables",
            "🤖 ML Comparison",
        ])

        # OVERVIEW
        with sub_overview:
            st.markdown("#### Population Health Shifts (Weighted Prevalence)")
            col_a, col_b = st.columns(2)
            with col_a:
                fig_covid_path = FIGURES_DIR / "phase3_covid/fig_covid_impact_summary.png"
                if fig_covid_path.exists():
                    st.image(str(fig_covid_path), use_container_width=True)
                    st.caption("COVID impact summary: prevalence shifts, OR changes, and wellness variable hypothesis results.")
                else:
                    st.warning("Run `python src/17_comparison_figures.py` to generate this figure.")

            with col_b:
                fig_prev_path = FIGURES_DIR / "phase3_covid/fig_prevalence_comparison.png"
                if fig_prev_path.exists():
                    st.image(str(fig_prev_path), use_container_width=True)
                    st.caption("Weighted prevalence of ALT outcomes and wellness variables in both cohorts.")

            # Key numbers table
            prev_path = REPORTS_P3_DIR / "prevalence_comparison.csv"
            if prev_path.exists():
                st.markdown("#### Key Numbers: What Changed")
                prev_df = _pd.read_csv(prev_path)
                key_labels = [
                    "ALT > 40 U/L (all)", "Diabetes", "Obesity (BMI≥30)", "Central obesity",
                    "Depression (PHQ-9 ≥10)", "High sedentary (≥8h/day)", "Short sleep (<7h)",
                ]
                pivot = prev_df[prev_df["label"].isin(key_labels)].pivot_table(
                    index="label", columns="cohort", values="prev_pct"
                ).reset_index()
                if "2017-2018" in pivot.columns and "2021-2023" in pivot.columns:
                    pivot["_delta"] = (pivot["2021-2023"] - pivot["2017-2018"]).round(1)
                    pivot["Change"] = pivot["_delta"].apply(
                        lambda d: f"▲ +{abs(d):.1f} pp" if d > 1 else (f"▼ -{abs(d):.1f} pp" if d < -1 else f"→ ±{abs(d):.1f} pp")
                    )
                    pivot = pivot.drop(columns=["_delta"]).rename(columns={
                        "label": "Variable",
                        "2017-2018": "2017-2018 (%)",
                        "2021-2023": "2021-2023 (%)",
                    })
                    st.dataframe(pivot, use_container_width=True, hide_index=True)
                    st.caption("Weighted post-exclusion prevalence. ▲ = increased post-COVID  ▼ = decreased.")

        # MODEL COMPARISON
        with sub_model:
            st.markdown("#### Same 10-Predictor Model — Both Cohorts")
            st.markdown(
                "The same survey-weighted logistic regression was applied to both cohorts with identical "
                "predictors. Changes in ORs and significance reflect true population-level differences."
            )

            fig_forest_path = FIGURES_DIR / "phase3_covid/fig_model_comparison_forest.png"
            if fig_forest_path.exists():
                st.image(str(fig_forest_path), use_container_width=True)
                st.caption("Forest plot: ◆ = p<0.05 significant. Same predictor in both cohorts shown side by side.")
            else:
                st.warning("Run `python src/15_model_comparison.py` to generate this figure.")

            mc_path = REPORTS_P3_DIR / "model_comparison_combined.csv"
            if mc_path.exists():
                st.markdown("#### OR Comparison Table")
                mc = _pd.read_csv(mc_path)
                display = mc[[
                    "label",
                    "OR_2017", "CI_low_2017", "CI_high_2017", "p_2017", "sig_2017",
                    "OR_2021", "CI_low_2021", "CI_high_2021", "p_2021", "sig_2021",
                ]].rename(columns={
                    "label": "Predictor",
                    "OR_2017": "OR (17)", "CI_low_2017": "CI lo (17)", "CI_high_2017": "CI hi (17)",
                    "p_2017": "p (17)", "sig_2017": "★ (17)",
                    "OR_2021": "OR (21)", "CI_low_2021": "CI lo (21)", "CI_high_2021": "CI hi (21)",
                    "p_2021": "p (21)", "sig_2021": "★ (21)",
                })
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.markdown("""
                **Key takeaways:**
                - **Male sex** became highly significant post-COVID: OR 2.03 → **2.93**, p 0.065 → **0.0006**
                - **Triglycerides** effect strengthened: OR 1.77 → **1.97**, p = 0.014
                - **Diabetes** lost significance: OR 1.98 → 0.90 — possibly mediated by improved metabolic management
                - **Waist circumference** stable and significant in both cohorts: ~OR 1.02 per cm
                - **Heavy metals** non-significant in both cohorts
                """)

        # WELLNESS VARIABLES
        with sub_wellness:
            st.markdown("#### COVID-Era Wellness Variables: Depression, Sleep, Sedentary Behavior")
            st.markdown(
                "We tested whether PHQ-9 depression, short sleep (<7h), and high sedentary time (≥8h/day) "
                "predict elevated ALT after adjusting for the 10-predictor model. These variables are "
                "available in **both** NHANES cycles with identical definitions."
            )

            col_w1, col_w2 = st.columns([1.2, 1])
            with col_w1:
                fig_well = FIGURES_DIR / "phase3_covid/fig_wellness_forest_plot.png"
                if fig_well.exists():
                    st.image(str(fig_well), use_container_width=True)
                    st.caption("Adjusted ORs for wellness variables — neither cohort shows significant associations.")

            with col_w2:
                wh_path = REPORTS_P3_DIR / "wellness_hypothesis_combined.csv"
                if wh_path.exists():
                    wh = _pd.read_csv(wh_path)
                    show_cols = [
                        "label",
                        "weighted_prev_pct_2017", "OR_adj_2017", "p_adj_2017",
                        "weighted_prev_pct_2021", "OR_adj_2021", "p_adj_2021",
                    ]
                    wh_show = wh[[c for c in show_cols if c in wh.columns]].rename(columns={
                        "label": "Variable",
                        "weighted_prev_pct_2017": "Prev% (17)",
                        "OR_adj_2017": "OR adj (17)",
                        "p_adj_2017": "p (17)",
                        "weighted_prev_pct_2021": "Prev% (21)",
                        "OR_adj_2021": "OR adj (21)",
                        "p_adj_2021": "p (21)",
                    })
                    st.dataframe(wh_show, use_container_width=True, hide_index=True)

            st.markdown("""
            **Key finding — Wellness Variables:**
            - **Depression prevalence rose +3.6 pp** post-COVID (8.2% → 11.8%) — but NOT a significant predictor of elevated ALT in either cohort (p > 0.46)
            - **High sedentary rose +3.4 pp** (30.2% → 33.6%) — NOT significant (p > 0.23)
            - **Short sleep declined -4.7 pp** (25.0% → 20.3%) — NOT significant (p > 0.68)

            **Interpretation:** COVID-era lifestyle disruptions increased in prevalence but do **not directly affect liver enzyme levels** once metabolic factors (waist, triglycerides) are controlled. The dominant pathway remains metabolic, not behavioral.
            """)

        # ML COMPARISON
        with sub_ml:
            st.markdown("#### XGBoost Model: Pre vs Post COVID")
            st.markdown(
                "The same XGBoost hyperparameters (from the 2017-2018 grid search) were applied to the "
                "2021-2023 cohort. Feature importance (SHAP) and ROC curves are compared below."
            )

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                fig_shap = FIGURES_DIR / "phase3_covid/fig_shap_comparison.png"
                if fig_shap.exists():
                    st.image(str(fig_shap), use_container_width=True)
                    st.caption("SHAP feature importance: AST and GGT dominate in both cohorts. Waist circumference dropped in post-COVID ranking.")
            with col_m2:
                fig_roc = FIGURES_DIR / "phase3_covid/fig_roc_comparison.png"
                if fig_roc.exists():
                    st.image(str(fig_roc), use_container_width=True)
                    st.caption("ROC curves: both cohorts show high discriminatory ability. Slight decrease post-COVID (0.970 → 0.955).")

            # SHAP table
            shap_path = REPORTS_P3_DIR / "shap_comparison.csv"
            if shap_path.exists():
                st.markdown("#### Top 10 SHAP Features — Comparison")
                shap_df = _pd.read_csv(shap_path).head(10)
                shap_show = shap_df[["label", "shap_2017", "rank_2017", "shap_2021", "rank_2021", "rank_change"]].rename(columns={
                    "label": "Feature",
                    "shap_2017": "SHAP (17)",
                    "rank_2017": "Rank (17)",
                    "shap_2021": "SHAP (21)",
                    "rank_2021": "Rank (21)",
                    "rank_change": "Rank shift",
                })
                st.dataframe(shap_show, use_container_width=True, hide_index=True)

            st.markdown("""
            **Key ML findings:**
            - **AST and GGT dominate** in both cohorts — they are biochemically co-correlated with ALT, not causal
            - **Waist circumference dropped** from rank 6 → rank 21 in SHAP importance post-COVID
            - **AUC slightly lower** post-COVID (0.970 → 0.955) — the 2021-2023 cohort is marginally harder to classify
            - **Top 3 features identical** between cohorts: AST, GGT, Age — confirms model stability
            """)

    # ── TAB 3: GENERATE REPORT ────────────────────────────────────────────────
    with tab_report:
        st.markdown("### Generate a Health Risk Report")
        st.caption(
            "Describe a person's profile or a population group. The AI uses the updated 10-predictor model "
            "(including waist circumference and triglycerides) to write a structured report covering "
            "demographics, health conditions, risk assessment, and recommendations."
        )

        st.divider()

        # Status check
        if api_missing:
            st.error("ANTHROPIC_API_KEY not set. Export it and restart the app.")
        elif st.session_state.report_generator is None:
            st.warning("Report generator not initialized — try refreshing the page.")
        else:
            # Example buttons
            st.markdown("**Try an example:**")
            ex_cols = st.columns(2)
            for i, example in enumerate(REPORT_EXAMPLES):
                with ex_cols[i % 2]:
                    if st.button(example, key=f"ex_{hash(example)}"):
                        st.session_state._report_text = example
                        st.rerun()

            st.markdown("")

            # Text input — let Streamlit manage state via key, update on example click
            if "report_text_area" not in st.session_state:
                st.session_state.report_text_area = ""
            if st.session_state.get("_report_text"):
                st.session_state.report_text_area = st.session_state.pop("_report_text")

            report_input = st.text_area(
                "Describe the subject or topic:",
                height=100,
                placeholder=(
                    "Examples:\n"
                    "  52-year-old diabetic male, waist 104 cm, triglycerides 210, smoker\n"
                    "  Metabolic syndrome and liver injury risk in U.S. adults\n"
                    "  Hispanic women aged 40–60 with diabetes"
                ),
                key="report_text_area",
            )

            col_btn, col_clear = st.columns([1, 5])
            with col_btn:
                generate_clicked = st.button("Generate Report", type="primary", key="gen_btn")
            with col_clear:
                if st.button("Clear Report", key="clear_report"):
                    st.session_state.generated_report = None
                    st.session_state._report_text = ""
                    st.rerun()

            if generate_clicked:
                if not report_input.strip():
                    st.warning("Please describe a subject or topic first.")
                else:
                    st.session_state._report_text = report_input.strip()
                    with st.spinner("Generating report — this takes 20–40 seconds..."):
                        try:
                            report_md = st.session_state.report_generator.generate(report_input.strip())
                            st.session_state.generated_report = report_md
                        except Exception as e:
                            st.session_state.generated_report = None
                            st.error(f"Generation failed: {e}")

            # Display report
            if st.session_state.generated_report:
                st.divider()
                st.download_button(
                    label="⬇ Download Report (.md)",
                    data=st.session_state.generated_report,
                    file_name="nhanes_health_risk_report.md",
                    mime="text/markdown",
                )
                st.markdown(st.session_state.generated_report)


def _md_to_html(text: str) -> str:
    """Minimal markdown → HTML for the report output box."""
    import re
    lines = []
    for line in text.split("\n"):
        if line.startswith("### "):
            lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- ") or line.startswith("* "):
            lines.append(f"<li>{line[2:]}</li>")
        elif line.startswith("---"):
            lines.append("<hr style='border:none;border-top:1px solid #E6EAF0;margin:0.8rem 0;'>")
        elif line.strip() == "":
            lines.append("<br>")
        else:
            # Bold and italic
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            line = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', line)
            line = re.sub(r'`(.+?)`',       r'<code>\1</code>', line)
            lines.append(f"<p>{line}</p>")
    return "\n".join(lines)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        run_streamlit()
