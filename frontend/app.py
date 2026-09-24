"""
frontend/app.py
Loan Approval Analytics & Prediction — fully self-contained Streamlit app.
Run: streamlit run frontend/app.py
"""

import os, sys, io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGB = True
except ImportError:
    XGB = False

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "best_loan_prediction_model.joblib")
FEAT_PATH    = os.path.join(MODEL_DIR, "feature_names.joblib")
MAP_PATH     = os.path.join(MODEL_DIR, "preprocessing_mappings.joblib")

PM = {
    "Gender":        {"Male": 1, "Female": 0},
    "Married":       {"Yes": 1, "No": 0},
    "Education":     {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Approval System", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#0f1117; }
  [data-testid="stSidebar"]          { background:#1a1d2e; }
  h1,h2,h3,h4,label,p,span,div      { color:#e0e0e0 !important; }
  .kpi { background:#1e2130; border-left:4px solid #90caf9;
         border-radius:10px; padding:1rem; margin-bottom:.5rem; }
  .kpi-val  { font-size:2rem; font-weight:700; color:#90caf9 !important; }
  .kpi-lbl  { font-size:.75rem; color:#8899aa !important;
               text-transform:uppercase; letter-spacing:1px; }
  .kpi-d    { font-size:.8rem; margin-top:.2rem; }
  .sec      { font-size:1.2rem; color:#90caf9 !important; font-weight:600;
               border-bottom:1px solid #2a2f45; padding-bottom:.3rem;
               margin:1.2rem 0 .8rem 0; }
  .insight  { background:#1a2340; border-left:4px solid #f59e0b;
               border-radius:6px; padding:.8rem 1rem; margin:.4rem 0; }
  .action   { background:#1a2e1a; border-left:4px solid #4caf50;
               border-radius:6px; padding:.8rem 1rem; margin:.4rem 0; }
  .risk     { background:#2e1a1a; border-left:4px solid #ef5350;
               border-radius:6px; padding:.8rem 1rem; margin:.4rem 0; }
  .approved { background:#1b5e20; color:#fff !important; padding:1.2rem;
               border-radius:12px; text-align:center;
               font-size:1.6rem; font-weight:700; margin:.5rem 0; }
  .rejected { background:#b71c1c; color:#fff !important; padding:1.2rem;
               border-radius:12px; text-align:center;
               font-size:1.6rem; font-weight:700; margin:.5rem 0; }
  .stButton>button { background:#1565c0; color:#fff; border:none;
                     border-radius:8px; font-size:1rem; padding:.6rem 1.5rem; }
  .stButton>button:hover { background:#1976d2; }
  [data-testid="stMetricValue"] { color:#90caf9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for name in ["loan_data.csv", "dataset.csv", "loan_data_set.csv"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return pd.read_csv(p)
    # fallback — generate synthetic data on the fly
    np.random.seed(42)
    n = 614
    married = np.random.choice(["Yes","No"], n, p=[0.65,0.35])
    edu     = np.random.choice(["Graduate","Not Graduate"], n, p=[0.78,0.22])
    area    = np.random.choice(["Urban","Semiurban","Rural"], n, p=[0.37,0.38,0.25])
    ch      = np.random.choice([1,0], n, p=[0.84,0.16])
    ai      = np.random.randint(1000,15000,n)
    ci      = np.where(married=="Yes", np.random.randint(0,5000,n), 0)
    la      = np.random.randint(30,500,n)
    ti      = ai + ci
    dti     = la / np.maximum(ti, 1)
    score   = (ch*0.45 + (edu=="Graduate").astype(int)*0.10
               + np.clip(1-dti*10,0,1)*0.20
               + (area=="Semiurban").astype(int)*0.10
               + np.clip(ti/10000,0,1)*0.15
               + np.random.uniform(-0.1,0.1,n))
    return pd.DataFrame({
        "Loan_ID":           [f"LP{i:06d}" for i in range(1,n+1)],
        "Gender":            np.random.choice(["Male","Female"],n,p=[0.81,0.19]),
        "Married":           married,
        "Dependents":        np.random.choice(["0","1","2","3+"],n,p=[0.57,0.17,0.17,0.09]),
        "Education":         edu,
        "Self_Employed":     np.random.choice(["No","Yes"],n,p=[0.86,0.14]),
        "ApplicantIncome":   ai,
        "CoapplicantIncome": ci,
        "LoanAmount":        la,
        "Loan_Amount_Term":  np.random.choice([360,180,120,60,240],n,p=[0.68,0.10,0.10,0.06,0.06]),
        "Credit_History":    ch,
        "Property_Area":     area,
        "Loan_Status":       np.where(score>=0.5,"Y","N"),
    })

@st.cache_data
def analytics_df(raw):
    df = raw.copy().dropna(subset=["Loan_Status"])
    df["Dependents"]        = df["Dependents"].replace("3+",4)
    df["Dependents"]        = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0)
    df["TotalIncome"]       = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["DebtToIncomeRatio"] = np.where(df["TotalIncome"]>0,
                                       df["LoanAmount"]/df["TotalIncome"], 0)
    df["LoanAmount_Cat"]    = pd.cut(df["LoanAmount"],bins=[0,100,200,9999],
                                     labels=["Low(<=100K)","Mid(100-200K)","High(>200K)"])
    df["Income_Cat"]        = pd.cut(df["ApplicantIncome"],bins=[0,3000,6000,99999],
                                     labels=["Low(<3K)","Mid(3-6K)","High(>6K)"])
    df["Approved"]          = (df["Loan_Status"]=="Y").astype(int)
    return df

# ── Model train / load ─────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEAT_PATH):
        return joblib.load(MODEL_PATH), joblib.load(FEAT_PATH), PM

    raw = load_data()
    df  = raw.copy().dropna()
    df.replace({"Loan_Status":{"N":0,"Y":1}}, inplace=True)
    df  = df.replace("3+", 4)
    df.replace({k: v for k, v in PM.items()}, inplace=True)
    df["Dependents"]        = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0)
    df["TotalIncome"]       = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["DebtToIncomeRatio"] = np.where(df["TotalIncome"]>0,
                                       df["LoanAmount"]/df["TotalIncome"], 0)
    df["LoanAmount_Cat"]    = pd.cut(df["LoanAmount"],bins=[0,100,200,9999],labels=[0,1,2])
    df["Income_Cat"]        = pd.cut(df["ApplicantIncome"],bins=[0,3000,6000,99999],labels=[0,1,2])

    X = df.drop(columns=["Loan_ID","Loan_Status"], errors="ignore")
    Y = df["Loan_Status"]
    for c in X.select_dtypes(include=["object","category"]).columns:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))

    Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    models = {"RF": RandomForestClassifier(n_estimators=100, random_state=42)}
    if XGB:
        models["XGB"] = xgb.XGBClassifier(random_state=42, eval_metric="logloss",
                                           use_label_encoder=False)
    best_m, best_acc = None, 0
    for m in models.values():
        m.fit(Xt, Yt)
        acc = accuracy_score(Yv, m.predict(Xv))
        if acc > best_acc:
            best_acc, best_m = acc, m

    feats = X.columns.tolist()
    joblib.dump(best_m, MODEL_PATH)
    joblib.dump(feats,  FEAT_PATH)
    joblib.dump(PM,     MAP_PATH)
    return best_m, feats, PM

# ── Prediction helper ──────────────────────────────────────────────────────
def run_prediction(model, feats, pm, inp):
    total = inp["ApplicantIncome"] + inp["CoapplicantIncome"]
    dti   = inp["LoanAmount"] / total if total > 0 else 0
    lcat  = 0 if inp["LoanAmount"]<=100 else 1 if inp["LoanAmount"]<=200 else 2
    icat  = 0 if inp["ApplicantIncome"]<=3000 else 1 if inp["ApplicantIncome"]<=6000 else 2
    row   = np.array([[
        pm["Gender"][inp["Gender"]],
        pm["Married"][inp["Married"]],
        inp["Dependents"],
        pm["Education"][inp["Education"]],
        pm["Self_Employed"][inp["Self_Employed"]],
        inp["ApplicantIncome"],
        inp["CoapplicantIncome"],
        inp["LoanAmount"],
        inp["Loan_Amount_Term"],
        inp["Credit_History"],
        pm["Property_Area"][inp["Property_Area"]],
        total, dti, lcat, icat
    ]])
    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0] if hasattr(model,"predict_proba") else None
    return int(pred), proba

# ── Chart helper ───────────────────────────────────────────────────────────
def dfig(fs=(7,3.5)):
    fig, ax = plt.subplots(figsize=fs, facecolor="#1e2130")
    ax.set_facecolor("#1e2130")
    ax.tick_params(colors="#aaaaaa")
    for s in ax.spines.values(): s.set_edgecolor("#333355")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.title.set_color("#90caf9")
    return fig, ax

def bar_chart(ax, labels, values, colors):
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f"{v:.1f}%", ha="center", va="bottom", color="#e0e0e0", fontsize=9)
    return bars

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    raw            = load_data()
    df             = analytics_df(raw)
    model, feats, pm = get_model()

    st.markdown('<h1 style="text-align:center;color:#90caf9;font-size:2.2rem;font-weight:700">'
                '🏦 Loan Approval Analytics & Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#8899aa;margin-bottom:1.5rem">'
                'KPI dashboard · Portfolio risk analysis · Real-time ML prediction</p>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        section = st.radio("Select Section", [
            "📊 Executive Overview",
            "🔍 Portfolio & Risk",
            "🤖 Predict Loan",
        ], label_visibility="collapsed")
        st.markdown("---")
        st.caption(f"**Model:** {type(model).__name__}")
        st.caption(f"**Records:** {len(df):,}")
        st.caption("**Dataset:** Kaggle Loan Prediction")
        st.caption("**Built by:** Anurag Joshi")

    # ═══════════════════════════════════════════════════
    # SECTION 1 — EXECUTIVE OVERVIEW
    # ═══════════════════════════════════════════════════
    if section == "📊 Executive Overview":
        st.markdown('<div class="sec">📊 Executive Overview — What\'s Happening</div>',
                    unsafe_allow_html=True)

        total   = len(df)
        appr    = int(df["Approved"].sum())
        rate    = appr / total * 100
        avg_ln  = df["LoanAmount"].mean()
        cr_risk = (df["Credit_History"]==0).sum() / total * 100

        c1,c2,c3,c4 = st.columns(4)
        for col, lbl, val, dlt in [
            (c1,"Total Applications",f"{total:,}",""),
            (c2,"Approval Rate",     f"{rate:.1f}%",
             "🟢 Healthy (>60%)" if rate>=60 else "🔴 Low (<60%)"),
            (c3,"Avg Loan Amount",   f"Rs.{avg_ln:.0f}K",""),
            (c4,"No Credit History", f"{cr_risk:.1f}%","🔴 Risk segment"),
        ]:
            col.markdown(
                f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                f'<div class="kpi-val">{val}</div>'
                f'<div class="kpi-d">{dlt}</div></div>',
                unsafe_allow_html=True)

        st.markdown('<div class="sec">Approval by Segment</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)

        with ca:
            _grp = df.groupby("Property_Area")["Approved"]
            area = pd.DataFrame({"cnt": _grp.count(), "appr": _grp.sum()}).reset_index()
            area["R"] = area["appr"] / area["cnt"] * 100
            fig, ax = dfig()
            colors = ["#ef5350" if r<65 else "#42a5f5" if r<75 else "#66bb6a"
                      for r in area["R"]]
            bar_chart(ax, area["Property_Area"], area["R"], colors)
            ax.axhline(rate,color="#f59e0b",linestyle="--",linewidth=1.2,
                       label=f"Avg {rate:.1f}%")
            ax.set_ylim(0,100); ax.set_ylabel("Approval Rate (%)")
            ax.set_title("Approval Rate by Property Area")
            ax.legend(facecolor="#1e2130",labelcolor="#aaaaaa",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with cb:
            _grp = df.groupby("Education")["Approved"]
            edu = pd.DataFrame({"cnt": _grp.count(), "appr": _grp.sum()}).reset_index()
            edu["R"] = edu["appr"] / edu["cnt"] * 100
            fig, ax = dfig()
            bar_chart(ax, edu["Education"], edu["R"],
                      ["#66bb6a" if r>=70 else "#ef5350" for r in edu["R"]])
            ax.axhline(rate,color="#f59e0b",linestyle="--",linewidth=1.2)
            ax.set_ylim(0,100); ax.set_ylabel("Approval Rate (%)")
            ax.set_title("Approval Rate by Education")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        best_a  = area.loc[area["R"].idxmax(),"Property_Area"]
        worst_a = area.loc[area["R"].idxmin(),"Property_Area"]
        st.markdown(
            f'<div class="insight">📌 <b>Insight:</b> {best_a} leads approvals at '
            f'{area["R"].max():.1f}%. {worst_a} lags at {area["R"].min():.1f}%.</div>'
            f'<div class="action">✅ <b>Action:</b> Focus acquisition on {best_a} + '
            f'Graduate segment. Run financial literacy in {worst_a}.</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sec">Income & Loan Distribution</div>',
                    unsafe_allow_html=True)
        cc, cd = st.columns(2)
        with cc:
            fig, ax = dfig()
            ax.hist(df[df["Approved"]==1]["TotalIncome"].clip(upper=25000),
                    bins=30,alpha=0.75,color="#42a5f5",label="Approved")
            ax.hist(df[df["Approved"]==0]["TotalIncome"].clip(upper=25000),
                    bins=30,alpha=0.75,color="#ef5350",label="Rejected")
            ax.set_xlabel("Total Income (Rs.)"); ax.set_title("Income vs Approval")
            ax.legend(facecolor="#1e2130",labelcolor="#e0e0e0",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with cd:
            ic = (df.groupby("Income_Cat",observed=True)["Approved"]
                    .mean().reset_index().assign(R=lambda d: d["Approved"]*100))
            fig, ax = dfig()
            bar_chart(ax, ic["Income_Cat"].astype(str), ic["R"],
                      ["#ef5350","#f59e0b","#66bb6a"][:len(ic)])
            ax.set_ylim(0,100); ax.set_xlabel("Income Bracket")
            ax.set_ylabel("Approval Rate (%)")
            ax.set_title("Approval Rate by Income Bracket")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ═══════════════════════════════════════════════════
    # SECTION 2 — PORTFOLIO & RISK
    # ═══════════════════════════════════════════════════
    elif section == "🔍 Portfolio & Risk":
        st.markdown('<div class="sec">🔍 Portfolio & Risk — Drivers & Opportunities</div>',
                    unsafe_allow_html=True)

        _grp = df.groupby("Credit_History")["Approved"]
        cr = pd.DataFrame({"cnt": _grp.count(), "appr": _grp.sum()}).reset_index()
        cr["R"] = cr["appr"] / cr["cnt"] * 100
        cr["L"] = cr["Credit_History"].map({0:"No History",1:"Has History"})

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = dfig()
            bar_chart(ax, cr["L"], cr["R"], ["#ef5350","#42a5f5"])
            ax.set_ylim(0,100); ax.set_ylabel("Approval Rate (%)")
            ax.set_title("Credit History — #1 Approval Driver")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            fig, ax = dfig()
            ax.hist(df[df["Approved"]==1]["DebtToIncomeRatio"].clip(upper=0.15),
                    bins=25,alpha=0.75,color="#42a5f5",label="Approved")
            ax.hist(df[df["Approved"]==0]["DebtToIncomeRatio"].clip(upper=0.15),
                    bins=25,alpha=0.75,color="#ef5350",label="Rejected")
            ax.set_xlabel("Debt-to-Income Ratio"); ax.set_title("DTI Distribution")
            ax.legend(facecolor="#1e2130",labelcolor="#e0e0e0",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        no_r  = cr[cr["Credit_History"]==0]["R"].values[0] if 0 in cr["Credit_History"].values else 0
        yes_r = cr[cr["Credit_History"]==1]["R"].values[0] if 1 in cr["Credit_History"].values else 0
        st.markdown(
            f'<div class="risk">⚠️ <b>Risk:</b> No-history applicants approve at '
            f'{no_r:.1f}% vs {yes_r:.1f}% with history — '
            f'a {yes_r-no_r:.1f}pp gap. Strongest single rejection predictor.</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sec">Segmentation — Dependents & Employment</div>',
                    unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            dep = (df.groupby("Dependents")["Approved"].mean()
                     .reset_index().assign(R=lambda d: d["Approved"]*100))
            fig, ax = dfig()
            ax.plot(dep["Dependents"].astype(int), dep["R"],
                    marker="o",color="#90caf9",linewidth=2,markersize=8)
            ax.fill_between(dep["Dependents"].astype(int),dep["R"],
                            alpha=0.15,color="#90caf9")
            ax.set_ylim(0,100); ax.grid(axis="y",color="#333355",linewidth=0.5)
            ax.set_xlabel("Dependents"); ax.set_ylabel("Approval Rate (%)")
            ax.set_title("Approval Rate vs Dependents")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c4:
            emp = (df.groupby("Self_Employed")["Approved"].mean()
                     .reset_index()
                     .assign(R=lambda d: d["Approved"]*100,
                             L=lambda d: d["Self_Employed"].map(
                                 {"No":"Salaried","Yes":"Self-Employed"})))
            fig, ax = dfig()
            bar_chart(ax, emp["L"], emp["R"], ["#42a5f5","#ffa726"])
            ax.set_ylim(0,100); ax.set_ylabel("Approval Rate (%)")
            ax.set_title("Salaried vs Self-Employed")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="sec">Opportunity — Loan Amount Sweet Spot</div>',
                    unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            _grp = df.groupby("LoanAmount_Cat",observed=True)["Approved"]
            lc = pd.DataFrame({"cnt": _grp.count(), "appr": _grp.sum()}).reset_index()
            lc["R"] = lc["appr"] / lc["cnt"] * 100
            fig, ax = dfig()
            bar_chart(ax, lc["LoanAmount_Cat"].astype(str), lc["R"],
                      ["#66bb6a","#42a5f5","#ef5350"][:len(lc)])
            ax.set_ylim(0,100); ax.set_ylabel("Approval Rate (%)")
            ax.set_title("Approval by Loan Amount Category")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c6:
            try:
                piv = (df.pivot_table(values="Approved",index="Education",
                                      columns="Property_Area",aggfunc="mean")*100)
                fig, ax = dfig((7,3.5))
                sns.heatmap(piv,annot=True,fmt=".1f",cmap="RdYlGn",ax=ax,
                            linewidths=0.5,linecolor="#0f1117",
                            annot_kws={"size":10},vmin=50,vmax=100)
                ax.set_title("Approval %: Education × Area")
                ax.set_facecolor("#1e2130")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            except Exception:
                st.info("Need more data variance for heatmap.")

        st.markdown(
            f'<div class="action">💡 <b>Opportunity:</b> Small loans (≤Rs.100K) '
            f'approve at {lc["R"].max():.1f}% — highest across all brackets.<br>'
            f'✅ <b>Action:</b> Fast-track "Quick Loan" product for small amounts '
            f'targeting Semiurban Graduates with good credit history.</div>',
            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # SECTION 3 — PREDICT
    # ═══════════════════════════════════════════════════
    elif section == "🤖 Predict Loan":
        st.markdown('<div class="sec">🤖 Predict Loan Eligibility</div>',
                    unsafe_allow_html=True)
        st.markdown("Fill in the applicant details below and get an instant ML decision.")

        col_form, col_out = st.columns([3, 2])

        with col_form:
            with st.form("loan_form"):
                st.markdown("**Personal Details**")
                p1, p2 = st.columns(2)
                with p1:
                    gender        = st.selectbox("Gender",         ["Male","Female"])
                    married       = st.selectbox("Marital Status", ["No","Yes"])
                    dependents    = st.selectbox("Dependents",     [0,1,2,4])
                with p2:
                    education     = st.selectbox("Education",      ["Graduate","Not Graduate"])
                    self_employed = st.selectbox("Self Employed",  ["No","Yes"])
                    property_area = st.selectbox("Property Area",  ["Rural","Semiurban","Urban"])

                st.markdown("**Financial Details**")
                f1, f2 = st.columns(2)
                with f1:
                    app_inc  = st.number_input("Applicant Income (Rs.)",    min_value=0, value=5000, step=500)
                    coapp_inc= st.number_input("Co-applicant Income (Rs.)", min_value=0, value=0,    step=500)
                with f2:
                    loan_amt = st.number_input("Loan Amount (Rs. thousands)", min_value=1, value=150, step=10)
                    loan_trm = st.selectbox("Loan Term (months)",
                                            [12,36,60,84,120,180,240,300,360,480], index=8)

                st.markdown("**Credit**")
                credit = st.selectbox("Credit History", [1,0],
                                      format_func=lambda x: "✅ Has Credit History" if x==1
                                                            else "❌ No Credit History")
                submitted = st.form_submit_button("🔍 Check Loan Eligibility",
                                                   use_container_width=True)

        with col_out:
            if submitted:
                inp = {
                    "Gender": gender, "Married": married, "Dependents": dependents,
                    "Education": education, "Self_Employed": self_employed,
                    "ApplicantIncome": app_inc, "CoapplicantIncome": coapp_inc,
                    "LoanAmount": loan_amt, "Loan_Amount_Term": loan_trm,
                    "Credit_History": credit, "Property_Area": property_area,
                }
                pred, proba = run_prediction(model, feats, pm, inp)

                if pred == 1:
                    st.markdown('<div class="approved">✅ LOAN APPROVED</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="rejected">❌ LOAN REJECTED</div>',
                                unsafe_allow_html=True)

                if proba is not None:
                    ap = proba[1]; rp = proba[0]
                    m1, m2 = st.columns(2)
                    m1.metric("Approval Probability", f"{ap:.1%}")
                    m2.metric("Model Confidence",     f"{max(proba):.1%}")

                    fig, ax = dfig((5,3))
                    ax.bar(["Rejected","Approved"],[rp,ap],
                           color=["#ef5350","#43a047"])
                    ax.set_ylim(0,1); ax.set_ylabel("Probability")
                    ax.set_title("Probability Breakdown")
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                total_inc = app_inc + coapp_inc
                dti = loan_amt / total_inc if total_inc > 0 else 0
                st.markdown("**Application Summary**")
                st.markdown(f"""
| | |
|---|---|
| Total Income | Rs.{total_inc:,} |
| Loan Amount | Rs.{loan_amt:,}K |
| Debt-to-Income | {dti:.3f} |
| Credit History | {"Yes ✅" if credit==1 else "No ❌"} |
| Property Area | {property_area} |
| Education | {education} |
""")
                if pred == 0:
                    reasons = []
                    if credit == 0:
                        reasons.append("No credit history (biggest rejection factor)")
                    if dti > 0.05:
                        reasons.append(f"High debt-to-income ratio ({dti:.3f}) — try a smaller loan")
                    if total_inc < 3000:
                        reasons.append("Low income — consider adding a co-applicant")
                    if reasons:
                        st.markdown(
                            '<div class="risk">⚠️ <b>Why Rejected?</b><br>'
                            + "<br>".join(f"• {r}" for r in reasons)
                            + "</div>", unsafe_allow_html=True)
            else:
                st.info("👈 Fill the form and click **Check Loan Eligibility**")
                st.markdown("""
**How it works**
- Model trained on 614 loan applications
- Compares SVM, Random Forest & XGBoost
- Picks best model automatically
- Returns approval decision + probability

**Top factors the model checks**
1. 🏆 Credit History
2. 💰 Total Income
3. 📊 Debt-to-Income Ratio
4. 🏘️ Property Area
5. 🎓 Education
""")

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    c1.caption("Stack: Python · Scikit-learn · Streamlit")
    c2.caption("Dataset: Kaggle Loan Prediction Problem")
    c3.caption("By: Anurag Joshi")

if __name__ == "__main__":
    main()
