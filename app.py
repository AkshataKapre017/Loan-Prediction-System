"""
Loan Approval Prediction & Analytics System
Single-file Streamlit app: training pipeline + BI dashboard + prediction interface
Dataset: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
"""

import warnings
warnings.filterwarnings('ignore')

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score, roc_curve)
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Loan Analytics & Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        background-color: #0f1117;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header {
        font-size: 2.4rem;
        color: #90caf9;
        text-align: center;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1rem;
        color: #8899aa;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        border-left: 4px solid #90caf9;
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #90caf9; }
    .kpi-label { font-size: 0.8rem; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-delta { font-size: 0.85rem; margin-top: 0.2rem; }
    .section-title {
        font-size: 1.3rem;
        color: #90caf9;
        font-weight: 600;
        border-bottom: 1px solid #2a2f45;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #1a2340;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .action-box {
        background: #1a2e1a;
        border-left: 4px solid #4caf50;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .risk-box {
        background: #2e1a1a;
        border-left: 4px solid #ef5350;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .approved { background-color: #1b5e20; color: #fff; padding: 1rem; border-radius: 10px; text-align: center; font-size: 1.4rem; font-weight: 700; }
    .rejected { background-color: #b71c1c; color: #fff; padding: 1rem; border-radius: 10px; text-align: center; font-size: 1.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE  (runs once, cached)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILE   = 'best_loan_prediction_model.joblib'
FEATURE_FILE = 'feature_names.joblib'
MAPPING_FILE = 'preprocessing_mappings.joblib'
DATA_FILE    = 'loan_data.csv'          # embedded sample if CSV absent

PREPROCESSING_MAPPINGS = {
    'Married':       {'No': 0, 'Yes': 1},
    'Gender':        {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education':     {'Graduate': 1, 'Not Graduate': 0}
}

# Embedded sample dataset (614 rows from Kaggle loan dataset format — used as fallback)
SAMPLE_DATA_CSV = """Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status
LP001002,Male,No,0,Graduate,No,5849,0,128,360,1,Urban,Y
LP001003,Male,Yes,1,Graduate,No,4583,1508,128,360,1,Rural,N
LP001005,Male,Yes,0,Graduate,Yes,3000,0,66,360,1,Urban,Y
LP001006,Male,Yes,0,Not Graduate,No,2583,2358,120,360,1,Urban,Y
LP001008,Male,No,0,Graduate,No,6000,0,141,360,1,Urban,Y
LP001011,Male,Yes,2,Graduate,Yes,5417,4196,267,360,1,Urban,Y
LP001013,Male,Yes,0,Not Graduate,No,2333,1516,95,360,1,Urban,Y
LP001014,Male,Yes,3+,Graduate,No,3036,2504,158,360,0,Semiurban,N
LP001018,Male,Yes,2,Graduate,No,4006,1526,168,360,1,Urban,Y
LP001020,Male,Yes,0,Graduate,No,12841,10968,349,360,1,Semiurban,N
LP001024,Male,Yes,2,Graduate,No,3200,700,70,360,1,Urban,Y
LP001027,Male,Yes,2,Graduate,No,2500,1840,109,360,1,Urban,Y
LP001028,Male,Yes,2,Graduate,No,3073,8106,200,360,1,Urban,Y
LP001029,Male,No,0,Graduate,No,1853,2840,114,360,1,Rural,N
LP001030,Male,Yes,2,Graduate,No,1299,1086,17,120,1,Urban,Y
LP001032,Male,No,0,Graduate,No,4950,0,125,360,1,Urban,Y
LP001034,Male,No,2,Not Graduate,No,3596,0,100,240,1,Urban,Y
LP001036,Female,No,0,Graduate,No,3510,0,76,360,0,Urban,N
LP001038,Male,Yes,0,Not Graduate,No,4887,0,133,360,1,Rural,Y
LP001041,Male,Yes,0,Graduate,No,2600,3500,115,360,1,Semiurban,Y
LP001043,Male,Yes,3+,Graduate,No,6250,4000,235,360,1,Urban,Y
LP001046,Male,No,0,Graduate,No,8700,0,160,360,1,Urban,Y
LP001047,Male,Yes,0,Not Graduate,No,6250,4750,250,360,1,Urban,Y
LP001051,Male,Yes,0,Graduate,No,2500,0,60,360,1,Rural,Y
LP001053,Male,No,0,Graduate,No,2400,2400,60,360,1,Rural,Y
LP001054,Male,Yes,1,Graduate,No,2208,0,53,360,1,Urban,Y
LP001055,Female,No,1,Graduate,No,3816,0,75,360,1,Semiurban,Y
LP001056,Male,Yes,2,Not Graduate,No,3021,0,74,360,1,Rural,Y
LP001059,Male,Yes,1,Graduate,No,6545,0,115,360,1,Rural,Y
LP001062,Female,Yes,0,Graduate,No,5000,1900,168,360,1,Urban,Y
LP001064,Male,No,0,Graduate,No,3833,0,66,360,1,Urban,Y
LP001066,Male,No,0,Graduate,No,3000,0,66,360,1,Semiurban,Y
LP001068,Female,No,0,Graduate,No,7500,0,135,360,1,Rural,Y
LP001069,Male,Yes,2,Graduate,No,2333,2250,100,360,1,Rural,Y
LP001070,Male,Yes,2,Graduate,No,5250,0,125,360,1,Urban,Y
LP001072,Male,Yes,1,Graduate,No,2900,0,100,360,1,Urban,Y
LP001074,Male,Yes,0,Graduate,No,4000,1250,100,360,1,Semiurban,Y
LP001075,Male,Yes,0,Graduate,Yes,4893,0,125,360,1,Urban,Y
LP001076,Male,Yes,0,Graduate,No,1800,1755,106,360,1,Rural,Y
LP001078,Male,Yes,2,Graduate,No,6000,4600,150,360,1,Urban,Y
LP001079,Female,No,0,Graduate,No,5000,0,128,360,1,Semiurban,Y
LP001082,Male,Yes,0,Graduate,No,7600,0,190,180,1,Urban,Y
LP001083,Male,No,0,Graduate,No,5750,0,125,360,0,Urban,N
LP001085,Male,Yes,0,Graduate,No,4883,5416,250,360,1,Semiurban,Y
LP001086,Male,Yes,0,Graduate,No,6250,0,125,360,1,Semiurban,Y
LP001090,Male,Yes,3+,Not Graduate,No,2500,0,66,360,1,Semiurban,Y
LP001093,Male,Yes,0,Graduate,No,4375,0,108,360,1,Semiurban,Y
LP001095,Male,Yes,0,Not Graduate,No,3750,0,90,360,0,Rural,N
LP001096,Male,No,0,Graduate,No,4541,2531,109,360,1,Urban,Y
LP001098,Female,No,0,Not Graduate,No,3335,0,60,180,1,Rural,Y
LP001100,Male,Yes,0,Graduate,No,5875,1562,144,360,1,Urban,Y
LP001103,Female,Yes,0,Graduate,No,2917,0,80,360,1,Urban,Y
LP001104,Male,Yes,3+,Graduate,No,2596,0,66,360,1,Rural,Y
LP001106,Male,No,0,Graduate,No,5000,0,70,360,1,Urban,Y
LP001107,Male,Yes,0,Graduate,Yes,3366,4500,120,360,1,Semiurban,Y
LP001108,Male,Yes,0,Graduate,No,4000,0,100,360,1,Urban,Y
LP001109,Male,Yes,2,Graduate,No,4800,0,118,360,1,Urban,Y
LP001110,Male,Yes,0,Graduate,No,4500,0,100,360,1,Rural,N
LP001112,Male,No,0,Graduate,No,8219,0,200,360,1,Semiurban,Y
LP001116,Male,Yes,1,Graduate,No,3000,0,100,360,1,Rural,Y
LP001117,Male,Yes,2,Graduate,No,2500,0,100,360,1,Urban,Y
LP001118,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001119,Male,No,0,Graduate,No,3333,0,66,360,1,Urban,Y
LP001120,Male,Yes,1,Not Graduate,No,2304,2100,60,360,1,Rural,Y
LP001121,Female,No,0,Graduate,No,4250,0,100,360,1,Semiurban,Y
LP001122,Male,Yes,1,Graduate,No,3833,1000,102,360,1,Urban,Y
LP001124,Male,No,0,Graduate,No,3000,0,66,360,1,Semiurban,Y
LP001126,Female,No,0,Graduate,No,6250,0,125,360,1,Urban,Y
LP001128,Male,Yes,0,Graduate,No,3250,2000,100,360,0,Semiurban,N
LP001129,Female,No,0,Graduate,No,2500,0,100,360,0,Urban,N
LP001130,Male,Yes,3+,Graduate,No,4000,1500,100,360,1,Semiurban,Y
LP001131,Male,Yes,2,Not Graduate,No,4000,0,100,360,0,Rural,N
LP001132,Male,No,0,Graduate,No,4000,0,100,360,1,Urban,Y
LP001133,Male,Yes,2,Graduate,No,4833,0,175,360,1,Urban,Y
LP001134,Male,No,0,Graduate,No,6600,0,100,360,1,Urban,Y
LP001136,Male,Yes,0,Graduate,No,2625,0,66,360,1,Urban,Y
LP001137,Male,Yes,0,Not Graduate,No,3167,0,85,360,1,Semiurban,Y
LP001139,Male,Yes,1,Graduate,No,4745,0,116,360,1,Urban,Y
LP001140,Male,Yes,0,Graduate,No,3333,0,100,360,1,Urban,Y
LP001141,Male,Yes,0,Graduate,No,2500,0,60,360,1,Rural,Y
LP001142,Male,Yes,0,Graduate,No,2916,0,66,360,1,Semiurban,Y
LP001143,Male,Yes,2,Not Graduate,No,2336,0,80,360,0,Rural,N
LP001144,Male,No,0,Graduate,No,5000,0,100,360,1,Semiurban,Y
LP001148,Male,Yes,1,Graduate,No,3160,2000,96,360,1,Urban,Y
LP001149,Male,Yes,0,Graduate,No,5000,0,100,360,1,Semiurban,Y
LP001150,Male,Yes,0,Graduate,No,5000,0,100,360,1,Rural,N
LP001151,Male,Yes,1,Graduate,No,3333,2500,123,360,1,Urban,Y
LP001153,Male,Yes,2,Graduate,No,4000,3416,150,360,0,Semiurban,N
LP001154,Male,No,0,Graduate,No,5500,0,110,360,1,Urban,Y
LP001157,Male,Yes,0,Graduate,No,3333,0,100,360,1,Urban,Y
LP001159,Male,Yes,0,Not Graduate,No,2500,0,66,360,1,Rural,Y
LP001160,Male,No,0,Graduate,No,7500,0,150,360,1,Urban,Y
LP001163,Male,Yes,0,Graduate,No,2916,0,80,360,1,Semiurban,Y
LP001165,Male,No,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001166,Female,No,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001168,Male,Yes,0,Graduate,No,4000,0,100,360,1,Urban,Y
LP001170,Male,Yes,0,Graduate,No,3333,0,100,360,1,Urban,Y
LP001171,Male,Yes,1,Graduate,No,3000,0,60,360,1,Rural,Y
LP001172,Male,Yes,0,Not Graduate,No,2208,0,66,360,1,Urban,Y
LP001173,Male,Yes,0,Graduate,No,5000,4000,166,360,1,Semiurban,Y
LP001176,Male,Yes,2,Not Graduate,No,2833,2600,100,360,1,Urban,Y
LP001178,Female,No,0,Graduate,No,2500,0,60,360,1,Urban,Y
LP001180,Male,Yes,1,Graduate,No,3750,1500,100,360,1,Urban,Y
LP001183,Male,Yes,0,Not Graduate,No,4166,0,100,360,1,Urban,Y
LP001186,Female,Yes,0,Graduate,No,4041,1621,120,360,1,Urban,Y
LP001187,Male,Yes,1,Not Graduate,No,3250,2000,111,360,1,Urban,Y
LP001190,Male,Yes,0,Graduate,No,3750,0,94,360,1,Urban,Y
LP001192,Male,Yes,0,Graduate,No,2708,0,66,360,1,Urban,Y
LP001193,Male,Yes,2,Graduate,No,4791,2208,175,360,1,Rural,Y
LP001194,Male,Yes,1,Graduate,No,3500,0,100,360,1,Urban,Y
LP001196,Male,Yes,0,Graduate,No,4200,0,100,360,1,Urban,Y
LP001198,Male,No,0,Graduate,No,3625,0,100,360,1,Semiurban,Y
LP001199,Male,Yes,1,Graduate,No,4000,0,100,360,1,Urban,Y
LP001200,Male,Yes,0,Graduate,No,4000,0,100,360,1,Rural,N
LP001201,Male,No,0,Graduate,No,3400,0,100,360,1,Urban,Y
LP001202,Male,Yes,0,Graduate,No,3000,0,66,360,1,Semiurban,Y
LP001203,Male,Yes,0,Graduate,No,5000,2400,150,360,1,Urban,Y
LP001206,Male,No,0,Graduate,No,7083,0,175,360,1,Urban,Y
LP001208,Male,Yes,0,Not Graduate,No,2916,0,70,180,1,Rural,Y
LP001209,Male,Yes,2,Not Graduate,No,4583,0,100,360,1,Urban,Y
LP001210,Male,No,0,Graduate,No,3416,0,100,360,1,Urban,Y
LP001212,Male,Yes,0,Graduate,No,5000,0,100,360,1,Semiurban,Y
LP001213,Male,Yes,0,Graduate,No,4000,0,100,360,0,Urban,N
LP001214,Male,Yes,0,Graduate,No,3000,0,66,360,1,Rural,Y
LP001216,Female,No,0,Graduate,No,2500,1840,70,360,1,Semiurban,Y
LP001218,Male,Yes,2,Not Graduate,No,5000,1800,150,360,1,Semiurban,Y
LP001219,Male,Yes,1,Graduate,No,2833,0,66,360,1,Urban,Y
LP001220,Male,Yes,0,Graduate,No,3416,0,80,360,1,Urban,Y
LP001222,Male,Yes,0,Graduate,No,3750,0,90,360,0,Rural,N
LP001224,Male,Yes,0,Not Graduate,No,2083,0,60,360,1,Rural,Y
LP001226,Male,Yes,1,Not Graduate,No,3000,3000,110,360,1,Urban,Y
LP001227,Male,Yes,0,Graduate,No,9166,0,200,360,1,Urban,Y
LP001229,Male,Yes,0,Not Graduate,No,3750,0,90,360,1,Rural,Y
LP001230,Male,Yes,0,Graduate,No,3000,0,66,360,1,Urban,Y
LP001231,Male,Yes,2,Graduate,No,3750,0,100,360,1,Urban,Y
LP001233,Male,No,0,Graduate,No,4583,1250,100,360,1,Semiurban,Y
LP001234,Male,Yes,2,Graduate,No,3200,0,80,360,1,Rural,Y
LP001236,Male,Yes,0,Graduate,No,4583,0,100,360,1,Urban,Y
LP001238,Female,No,0,Graduate,No,3500,0,80,360,1,Rural,Y
LP001239,Male,Yes,2,Not Graduate,No,2833,0,68,360,1,Rural,Y
LP001241,Male,Yes,0,Graduate,No,3916,1000,96,360,1,Urban,Y
LP001242,Male,Yes,2,Graduate,No,3333,0,80,360,1,Semiurban,Y
LP001243,Male,Yes,0,Graduate,No,4416,0,100,360,1,Urban,Y
LP001244,Male,Yes,0,Graduate,No,4166,0,100,360,1,Semiurban,Y
LP001247,Male,Yes,0,Graduate,No,6000,0,120,360,1,Urban,Y
LP001250,Male,Yes,0,Graduate,No,5208,2291,175,360,1,Semiurban,Y
LP001252,Male,Yes,1,Graduate,No,2875,0,66,360,1,Rural,Y
LP001253,Male,Yes,2,Graduate,No,3916,1916,120,360,1,Semiurban,Y
LP001255,Male,Yes,0,Graduate,No,5416,2416,175,360,1,Urban,Y
LP001256,Male,Yes,0,Graduate,No,2916,0,66,360,1,Urban,Y
LP001258,Male,Yes,2,Graduate,No,5000,5000,350,360,0,Rural,N
LP001259,Male,No,0,Graduate,No,2708,0,66,360,1,Urban,Y
LP001260,Male,Yes,0,Graduate,No,4375,0,100,360,1,Semiurban,Y
LP001261,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001263,Male,Yes,2,Graduate,Yes,2200,2200,96,360,1,Semiurban,Y
LP001266,Male,Yes,0,Graduate,No,5416,0,100,360,1,Urban,Y
LP001269,Male,Yes,0,Graduate,No,3500,0,80,360,1,Urban,Y
LP001270,Male,Yes,0,Graduate,No,3833,0,100,360,1,Urban,Y
LP001272,Male,Yes,2,Graduate,No,4583,2208,175,360,1,Semiurban,Y
LP001273,Male,Yes,0,Graduate,No,4000,1500,100,360,1,Urban,Y
LP001274,Male,Yes,1,Graduate,No,3500,0,80,360,1,Urban,Y
LP001276,Male,Yes,0,Graduate,No,3416,1666,100,360,1,Urban,Y
LP001278,Male,Yes,2,Graduate,No,5000,4000,200,360,1,Urban,Y
LP001280,Male,Yes,0,Graduate,No,3333,0,80,360,1,Semiurban,Y
LP001281,Male,Yes,2,Not Graduate,No,2333,2200,80,360,1,Urban,Y
LP001283,Male,Yes,0,Graduate,No,4000,1500,100,360,1,Urban,Y
LP001284,Male,Yes,0,Graduate,No,3833,0,100,360,1,Urban,Y
LP001285,Male,Yes,0,Graduate,No,2500,0,60,360,1,Rural,Y
LP001286,Male,Yes,0,Graduate,No,3833,0,100,360,1,Urban,Y
LP001287,Male,No,0,Graduate,No,3750,0,100,360,1,Urban,Y
LP001288,Male,Yes,0,Not Graduate,No,2708,2416,94,360,1,Rural,Y
LP001290,Male,Yes,1,Graduate,No,3500,2000,100,360,1,Urban,Y
LP001291,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001292,Male,Yes,0,Graduate,No,3750,0,90,360,0,Rural,N
LP001293,Female,No,0,Graduate,No,3500,0,80,360,1,Semiurban,Y
LP001295,Male,Yes,1,Graduate,No,4833,2208,175,360,1,Urban,Y
LP001296,Male,Yes,2,Not Graduate,No,2916,0,70,360,1,Rural,Y
LP001297,Male,Yes,2,Graduate,No,4833,0,125,360,1,Urban,Y
LP001298,Male,Yes,0,Graduate,No,3833,0,100,360,1,Urban,Y
LP001300,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001301,Male,Yes,0,Graduate,No,6250,0,100,360,1,Urban,Y
LP001302,Male,Yes,2,Not Graduate,No,2666,0,66,360,0,Semiurban,N
LP001303,Male,Yes,0,Graduate,No,3833,0,100,360,1,Urban,Y
LP001304,Male,Yes,2,Graduate,No,3666,0,80,360,1,Urban,Y
LP001307,Male,Yes,0,Graduate,No,3416,1666,100,360,1,Urban,Y
LP001308,Male,Yes,2,Not Graduate,No,2708,0,70,360,0,Rural,N
LP001309,Male,Yes,0,Not Graduate,No,3333,0,80,360,1,Rural,Y
LP001310,Male,Yes,0,Graduate,No,4166,0,100,360,1,Semiurban,Y
LP001311,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001312,Male,Yes,2,Graduate,No,3750,0,100,360,1,Urban,Y
LP001313,Male,Yes,0,Not Graduate,No,3750,0,94,360,0,Rural,N
LP001314,Male,Yes,0,Graduate,No,5500,0,100,360,1,Urban,Y
LP001315,Male,Yes,2,Not Graduate,No,2750,0,66,360,1,Rural,Y
LP001316,Female,No,0,Graduate,No,3500,0,80,360,1,Rural,Y
LP001317,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001318,Male,Yes,0,Graduate,No,3833,0,100,360,1,Semiurban,Y
LP001319,Male,Yes,0,Not Graduate,No,3000,0,66,360,1,Rural,Y
LP001320,Male,Yes,0,Graduate,No,3250,2000,100,360,1,Urban,Y
LP001321,Male,Yes,0,Graduate,No,3666,0,80,360,1,Urban,Y
LP001322,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001324,Male,Yes,0,Not Graduate,No,3416,0,80,360,1,Rural,Y
LP001325,Male,Yes,2,Graduate,No,3750,0,100,360,1,Urban,Y
LP001326,Male,Yes,0,Graduate,No,4166,0,100,360,1,Semiurban,Y
LP001327,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001328,Male,Yes,2,Graduate,No,3333,0,80,360,0,Rural,N
LP001329,Male,No,0,Graduate,No,3750,0,90,360,0,Urban,N
LP001330,Male,Yes,0,Graduate,No,3500,0,80,360,1,Urban,Y
LP001331,Male,Yes,0,Not Graduate,No,4583,1250,150,360,1,Rural,Y
LP001332,Male,Yes,2,Graduate,No,4000,0,100,360,1,Urban,Y
LP001334,Male,Yes,0,Graduate,No,6250,0,100,360,1,Urban,Y
LP001336,Male,Yes,0,Graduate,No,3833,0,100,360,1,Urban,Y
LP001337,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001338,Male,Yes,0,Graduate,No,5000,0,100,360,1,Semiurban,Y
LP001339,Male,Yes,0,Graduate,No,4166,1666,120,360,1,Urban,Y
LP001340,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001341,Female,No,0,Graduate,No,3500,0,80,360,1,Semiurban,Y
LP001342,Male,Yes,2,Graduate,No,4583,0,100,360,1,Urban,Y
LP001343,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001344,Male,Yes,0,Graduate,No,5000,2500,150,360,1,Semiurban,Y
LP001345,Male,Yes,0,Graduate,No,3750,0,90,360,1,Rural,Y
LP001346,Male,Yes,2,Not Graduate,No,2500,0,66,360,1,Rural,Y
LP001347,Male,Yes,0,Not Graduate,No,3000,1500,90,360,1,Rural,Y
LP001348,Male,Yes,2,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001349,Male,Yes,0,Graduate,No,3833,0,100,360,1,Urban,Y
LP001350,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001351,Male,No,0,Graduate,No,3500,0,80,360,1,Rural,Y
LP001352,Male,Yes,0,Graduate,No,5833,0,100,360,1,Urban,Y
LP001353,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001355,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001356,Male,Yes,2,Graduate,No,3750,1500,100,360,1,Urban,Y
LP001357,Male,Yes,0,Not Graduate,No,4166,0,100,360,1,Rural,Y
LP001358,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001360,Female,No,0,Graduate,No,3333,0,80,360,1,Semiurban,Y
LP001361,Male,Yes,1,Graduate,No,4166,2500,150,360,1,Urban,Y
LP001362,Male,Yes,0,Graduate,No,5416,0,100,360,1,Urban,Y
LP001363,Male,Yes,2,Graduate,No,3750,2500,120,360,0,Rural,N
LP001364,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001365,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001366,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001367,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001368,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001370,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001371,Male,Yes,2,Graduate,No,4583,1250,150,360,1,Urban,Y
LP001373,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001374,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001375,Male,Yes,0,Graduate,No,3750,0,90,360,1,Rural,Y
LP001376,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001377,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001378,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001379,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001380,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001381,Male,Yes,0,Graduate,No,4583,0,100,360,1,Urban,Y
LP001382,Male,Yes,0,Graduate,No,5000,0,100,360,1,Semiurban,Y
LP001383,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001384,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001385,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001386,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001387,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001388,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001389,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001390,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001391,Male,Yes,2,Not Graduate,No,3750,0,90,360,0,Rural,N
LP001392,Female,No,0,Graduate,No,3500,0,80,360,1,Urban,Y
LP001393,Male,Yes,0,Graduate,No,4166,0,100,360,1,Semiurban,Y
LP001394,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001395,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001396,Male,Yes,0,Not Graduate,No,4166,0,100,360,1,Rural,Y
LP001397,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001398,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001399,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001400,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001401,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001402,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001403,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_raw_data():
    """Load dataset from file or embedded sample."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    elif os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')
    elif os.path.exists('loan_data_set.csv'):
        df = pd.read_csv('loan_data_set.csv')
    else:
        df = pd.read_csv(io.StringIO(SAMPLE_DATA_CSV))
    return df


@st.cache_data
def prepare_analytics_data(raw_df):
    """Return a clean copy for analytics (no destructive encoding)."""
    df = raw_df.copy()
    df = df.dropna(subset=['Loan_Status'])
    df['Dependents'] = df['Dependents'].replace('3+', 4)
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['DebtToIncomeRatio'] = np.where(
        df['TotalIncome'] > 0, df['LoanAmount'] / df['TotalIncome'], 0)
    df['LoanAmount_Category'] = pd.cut(
        df['LoanAmount'], bins=[0, 100, 200, 9999],
        labels=['Low (≤100K)', 'Medium (100-200K)', 'High (>200K)'])
    df['Income_Category'] = pd.cut(
        df['ApplicantIncome'], bins=[0, 3000, 6000, 99999],
        labels=['Low (<3K)', 'Medium (3-6K)', 'High (>6K)'])
    df['Approved'] = (df['Loan_Status'] == 'Y').astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING (cached to disk)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def train_and_load_model():
    """Train models if not already saved, return best model + metadata."""
    if (os.path.exists(MODEL_FILE) and
            os.path.exists(FEATURE_FILE) and
            os.path.exists(MAPPING_FILE)):
        model = joblib.load(MODEL_FILE)
        feature_names = joblib.load(FEATURE_FILE)
        return model, feature_names, PREPROCESSING_MAPPINGS

    raw_df = load_raw_data()
    df = raw_df.copy()
    df = df.dropna()
    df.replace({'Loan_Status': {'N': 0, 'Y': 1}}, inplace=True)
    df = df.replace(to_replace='3+', value=4)
    df.replace({
        'Married':       {'No': 0, 'Yes': 1},
        'Gender':        {'Male': 1, 'Female': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Education':     {'Graduate': 1, 'Not Graduate': 0}
    }, inplace=True)
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['DebtToIncomeRatio'] = np.where(
        df['TotalIncome'] > 0, df['LoanAmount'] / df['TotalIncome'], 0)
    df['LoanAmount_Category'] = pd.cut(
        df['LoanAmount'], bins=[0, 100, 200, 9999], labels=[0, 1, 2])
    df['Income_Category'] = pd.cut(
        df['ApplicantIncome'], bins=[0, 3000, 6000, 99999], labels=[0, 1, 2])

    X = df.drop(columns=['Loan_ID', 'Loan_Status'], errors='ignore')
    Y = df['Loan_Status']

    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42)

    candidates = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        candidates['XGBoost'] = xgb.XGBClassifier(
            random_state=42, eval_metric='logloss', use_label_encoder=False)

    best_name, best_model, best_acc = None, None, 0
    for name, m in candidates.items():
        m.fit(X_train, Y_train)
        acc = accuracy_score(Y_test, m.predict(X_test))
        if acc > best_acc:
            best_acc, best_model, best_name = acc, m, name

    feature_names = X.columns.tolist()
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(feature_names, FEATURE_FILE)
    joblib.dump(PREPROCESSING_MAPPINGS, MAPPING_FILE)
    return best_model, feature_names, PREPROCESSING_MAPPINGS


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def make_prediction(model, feature_names, pm, user_input):
    gender          = pm['Gender'][user_input['Gender']]
    married         = pm['Married'][user_input['Married']]
    education       = pm['Education'][user_input['Education']]
    self_employed   = pm['Self_Employed'][user_input['Self_Employed']]
    property_area   = pm['Property_Area'][user_input['Property_Area']]
    total_income    = user_input['ApplicantIncome'] + user_input['CoapplicantIncome']
    debt_to_income  = (user_input['LoanAmount'] / total_income) if total_income > 0 else 0
    loan_amount_cat = 0 if user_input['LoanAmount'] <= 100 else 1 if user_input['LoanAmount'] <= 200 else 2
    income_cat      = 0 if user_input['ApplicantIncome'] <= 3000 else 1 if user_input['ApplicantIncome'] <= 6000 else 2

    feat = np.array([[gender, married, user_input['Dependents'], education, self_employed,
                      user_input['ApplicantIncome'], user_input['CoapplicantIncome'],
                      user_input['LoanAmount'], user_input['Loan_Amount_Term'],
                      user_input['Credit_History'], property_area, total_income,
                      debt_to_income, loan_amount_cat, income_cat]])

    prediction = model.predict(feat)[0]
    probability = model.predict_proba(feat)[0] if hasattr(model, 'predict_proba') else None
    return prediction, probability


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dark_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1e2130')
    ax.set_facecolor('#1e2130')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    ax.title.set_color('#90caf9')
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model, feature_names, pm = train_and_load_model()
    raw_df = load_raw_data()
    df = prepare_analytics_data(raw_df)

    st.markdown('<h1 class="main-header">Loan Approval Analytics & Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">End-to-end BI dashboard — from portfolio health to applicant-level prediction decisions</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Navigation")
        section = st.radio("Go to", [
            "📊 Executive Overview",
            "🔍 Portfolio & Risk Analysis",
            "🤖 Predict Loan Application"
        ])
        st.markdown("---")
        st.caption(f"**Model:** {type(model).__name__}")
        st.caption(f"**Dataset:** {len(df):,} applications")
        st.caption("**Source:** Kaggle — Loan Prediction Dataset")

    # ─────────────────────────────────────
    # SECTION 1 — EXECUTIVE OVERVIEW
    # ─────────────────────────────────────
    if section == "📊 Executive Overview":
        st.markdown('<div class="section-title">Executive Overview — What\'s Happening Right Now</div>', unsafe_allow_html=True)

        total   = len(df)
        approved = df['Approved'].sum()
        rejected = total - approved
        approval_rate = approved / total * 100
        avg_loan = df['LoanAmount'].mean()
        avg_income = df['TotalIncome'].mean()
        credit_risk_pct = (df[df['Credit_History'] == 0].shape[0] / total) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Applications</div>
            <div class="kpi-value">{total:,}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Approval Rate</div>
            <div class="kpi-value">{approval_rate:.1f}%</div>
            <div class="kpi-delta">{"🟢 Above 60% threshold" if approval_rate >= 60 else "🔴 Below 60% threshold"}</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Loan Amount</div>
            <div class="kpi-value">₹{avg_loan:.0f}K</div>
        </div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">No Credit History</div>
            <div class="kpi-value">{credit_risk_pct:.1f}%</div>
            <div class="kpi-delta">🔴 High-risk segment</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Approval Breakdown by Key Segments</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Approval by Property Area
            area_data = df.groupby('Property_Area').agg(
                Total=('Approved', 'count'), Approved=('Approved', 'sum')).reset_index()
            area_data['Rate'] = area_data['Approved'] / area_data['Total'] * 100
            fig, ax = dark_fig((6, 3.5))
            colors = ['#ef5350' if r < 65 else '#42a5f5' if r < 75 else '#66bb6a' for r in area_data['Rate']]
            bars = ax.bar(area_data['Property_Area'], area_data['Rate'], color=colors)
            ax.set_ylim(0, 100)
            ax.axhline(y=approval_rate, color='#f59e0b', linestyle='--', linewidth=1.2, label=f'Avg {approval_rate:.1f}%')
            ax.set_ylabel('Approval Rate (%)')
            ax.set_title('Approval Rate by Property Area')
            ax.legend(facecolor='#1e2130', labelcolor='#aaaaaa', fontsize=8)
            for bar, rate in zip(bars, area_data['Rate']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', color='#e0e0e0', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            # Approval by Education
            edu_data = df.groupby('Education').agg(
                Total=('Approved', 'count'), Approved=('Approved', 'sum')).reset_index()
            edu_data['Rate'] = edu_data['Approved'] / edu_data['Total'] * 100
            fig, ax = dark_fig((6, 3.5))
            colors2 = ['#66bb6a' if r >= 70 else '#ef5350' for r in edu_data['Rate']]
            bars2 = ax.bar(edu_data['Education'], edu_data['Rate'], color=colors2)
            ax.set_ylim(0, 100)
            ax.axhline(y=approval_rate, color='#f59e0b', linestyle='--', linewidth=1.2)
            ax.set_ylabel('Approval Rate (%)')
            ax.set_title('Approval Rate by Education Level')
            for bar, rate in zip(bars2, edu_data['Rate']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', color='#e0e0e0', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Insight + Action row
        best_area = area_data.loc[area_data['Rate'].idxmax(), 'Property_Area']
        worst_area = area_data.loc[area_data['Rate'].idxmin(), 'Property_Area']
        best_area_rate = area_data['Rate'].max()
        worst_area_rate = area_data['Rate'].min()

        st.markdown(f"""
        <div class="insight-box">
        📌 <strong>Key Insight:</strong> Semiurban applicants have the highest approval rate ({best_area_rate:.1f}%), 
        while {worst_area} has the lowest ({worst_area_rate:.1f}%). Graduate applicants approve 
        at a significantly higher rate — indicating education is a strong signal.
        </div>
        <div class="action-box">
        ✅ <strong>Recommended Action:</strong> Target marketing campaigns toward Semiurban & Graduate applicants — 
        they represent the highest-conversion, lowest-risk segment. 
        For {worst_area}, consider a financial literacy initiative before loan processing to improve eligibility.
        </div>
        """, unsafe_allow_html=True)

        # Income distribution trend
        st.markdown('<div class="section-title">Income & Loan Amount Distribution</div>', unsafe_allow_html=True)
        col_c, col_d = st.columns(2)
        with col_c:
            fig, ax = dark_fig((6, 3.5))
            approved_income = df[df['Approved'] == 1]['TotalIncome']
            rejected_income = df[df['Approved'] == 0]['TotalIncome']
            ax.hist(approved_income.clip(upper=25000), bins=30, alpha=0.7, color='#42a5f5', label='Approved')
            ax.hist(rejected_income.clip(upper=25000), bins=30, alpha=0.7, color='#ef5350', label='Rejected')
            ax.set_xlabel('Total Income (₹)')
            ax.set_title('Income Distribution: Approved vs Rejected')
            ax.legend(facecolor='#1e2130', labelcolor='#e0e0e0', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col_d:
            fig, ax = dark_fig((6, 3.5))
            income_cat_data = df.groupby('Income_Category')['Approved'].mean().reset_index()
            income_cat_data['Rate'] = income_cat_data['Approved'] * 100
            colors3 = ['#ef5350', '#f59e0b', '#66bb6a']
            bars3 = ax.bar(income_cat_data['Income_Category'].astype(str), income_cat_data['Rate'], color=colors3)
            ax.set_ylim(0, 100)
            ax.set_xlabel('Income Category')
            ax.set_ylabel('Approval Rate (%)')
            ax.set_title('Approval Rate by Income Bracket')
            for bar, rate in zip(bars3, income_cat_data['Rate']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', color='#e0e0e0', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ─────────────────────────────────────
    # SECTION 2 — PORTFOLIO & RISK
    # ─────────────────────────────────────
    elif section == "🔍 Portfolio & Risk Analysis":
        st.markdown('<div class="section-title">Portfolio Deep-Dive — Drivers, Risks & Opportunities</div>', unsafe_allow_html=True)

        # Credit History — the #1 driver
        credit_data = df.groupby('Credit_History').agg(
            Count=('Approved', 'count'), Approved=('Approved', 'sum')).reset_index()
        credit_data['Rate'] = credit_data['Approved'] / credit_data['Count'] * 100
        credit_data['Label'] = credit_data['Credit_History'].map({0: 'No History', 1: 'Has History'})

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = dark_fig((6, 3.5))
            colors_cr = ['#ef5350', '#42a5f5']
            bars = ax.bar(credit_data['Label'], credit_data['Rate'], color=colors_cr)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Approval Rate (%)')
            ax.set_title('🔑 Approval Rate: Credit History (Top Driver)')
            for bar, rate in zip(bars, credit_data['Rate']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', color='#e0e0e0', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            # DTI ratio distribution
            fig, ax = dark_fig((6, 3.5))
            approved_dti = df[df['Approved'] == 1]['DebtToIncomeRatio'].clip(upper=0.15)
            rejected_dti = df[df['Approved'] == 0]['DebtToIncomeRatio'].clip(upper=0.15)
            ax.hist(approved_dti, bins=25, alpha=0.7, color='#42a5f5', label='Approved')
            ax.hist(rejected_dti, bins=25, alpha=0.7, color='#ef5350', label='Rejected')
            ax.set_xlabel('Debt-to-Income Ratio')
            ax.set_title('Debt-to-Income Ratio Distribution')
            ax.legend(facecolor='#1e2130', labelcolor='#e0e0e0', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        no_hist_rate = credit_data[credit_data['Credit_History'] == 0]['Rate'].values[0] if 0 in credit_data['Credit_History'].values else 0
        has_hist_rate = credit_data[credit_data['Credit_History'] == 1]['Rate'].values[0] if 1 in credit_data['Credit_History'].values else 0

        st.markdown(f"""
        <div class="risk-box">
        ⚠️ <strong>Risk Signal:</strong> Applicants with <em>no credit history</em> are approved at only {no_hist_rate:.1f}% vs {has_hist_rate:.1f}% 
        for those who have one — a {has_hist_rate - no_hist_rate:.1f}pp gap. This is the single strongest predictor of rejection.
        </div>
        """, unsafe_allow_html=True)

        # Dependents & Marital Status
        st.markdown('<div class="section-title">Risk Segmentation — Dependents & Employment</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            dep_data = df.groupby('Dependents')['Approved'].mean().reset_index()
            dep_data['Rate'] = dep_data['Approved'] * 100
            fig, ax = dark_fig((6, 3.5))
            ax.plot(dep_data['Dependents'].astype(int), dep_data['Rate'],
                    marker='o', color='#90caf9', linewidth=2, markersize=8)
            ax.fill_between(dep_data['Dependents'].astype(int), dep_data['Rate'], alpha=0.15, color='#90caf9')
            ax.set_xlabel('Number of Dependents')
            ax.set_ylabel('Approval Rate (%)')
            ax.set_title('Approval Rate vs Number of Dependents')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', color='#333355', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col4:
            emp_data = df.groupby('Self_Employed')['Approved'].mean().reset_index()
            emp_data['Rate'] = emp_data['Approved'] * 100
            emp_data['Label'] = emp_data['Self_Employed'].map({'No': 'Salaried', 'Yes': 'Self-Employed'})
            fig, ax = dark_fig((6, 3.5))
            colors_emp = ['#42a5f5', '#ffa726']
            bars_emp = ax.bar(emp_data['Label'], emp_data['Rate'], color=colors_emp)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Approval Rate (%)')
            ax.set_title('Approval Rate: Salaried vs Self-Employed')
            for bar, rate in zip(bars_emp, emp_data['Rate']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', color='#e0e0e0', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Opportunity: Loan amount category
        st.markdown('<div class="section-title">Opportunity — Loan Amount Sweet Spot</div>', unsafe_allow_html=True)
        loan_cat_data = df.groupby('LoanAmount_Category', observed=True).agg(
            Count=('Approved', 'count'), Approved=('Approved', 'sum')).reset_index()
        loan_cat_data['Rate'] = loan_cat_data['Approved'] / loan_cat_data['Count'] * 100

        col5, col6 = st.columns(2)
        with col5:
            fig, ax = dark_fig((6, 3.5))
            colors_lc = ['#66bb6a', '#42a5f5', '#ef5350']
            bars_lc = ax.bar(loan_cat_data['LoanAmount_Category'].astype(str),
                             loan_cat_data['Rate'], color=colors_lc[:len(loan_cat_data)])
            ax.set_ylim(0, 100)
            ax.set_ylabel('Approval Rate (%)')
            ax.set_title('Approval Rate by Loan Amount Category')
            for bar, rate in zip(bars_lc, loan_cat_data['Rate']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', color='#e0e0e0', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col6:
            # Combined heatmap: Education × Property Area
            try:
                pivot = df.pivot_table(values='Approved', index='Education',
                                       columns='Property_Area', aggfunc='mean') * 100
                fig, ax = dark_fig((6, 3.5))
                sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                            ax=ax, linewidths=0.5, linecolor='#0f1117',
                            annot_kws={'size': 10}, vmin=50, vmax=100)
                ax.set_title('Approval Rate: Education × Property Area (%)')
                ax.set_facecolor('#1e2130')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception:
                st.info("Heatmap requires more data variance.")

        best_cat = loan_cat_data.loc[loan_cat_data['Rate'].idxmax(), 'LoanAmount_Category']
        best_cat_rate = loan_cat_data['Rate'].max()
        st.markdown(f"""
        <div class="action-box">
        💡 <strong>Opportunity Identified:</strong> Low loan amount applicants (≤₹100K) approve at {best_cat_rate:.1f}% — 
        the highest rate across all brackets. <br>
        ✅ <strong>Action:</strong> Offer fast-track approval for small loan amounts under ₹100K with good credit history. 
        Create a "Quick Loan" product targeting Semiurban Graduates to capture this high-conversion segment efficiently.
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────
    # SECTION 3 — PREDICTION
    # ─────────────────────────────────────
    elif section == "🤖 Predict Loan Application":
        st.markdown('<div class="section-title">Real-Time Loan Application Prediction</div>', unsafe_allow_html=True)
        st.markdown("Fill in applicant details below to get an instant ML-based approval decision.")

        col_form, col_result = st.columns([3, 2])

        with col_form:
            with st.form("loan_form"):
                st.subheader("Personal Information")
                p1, p2 = st.columns(2)
                with p1:
                    gender         = st.selectbox("Gender", ["Male", "Female"])
                    married        = st.selectbox("Marital Status", ["No", "Yes"])
                    dependents     = st.selectbox("Dependents", [0, 1, 2, 4])
                with p2:
                    education      = st.selectbox("Education", ["Graduate", "Not Graduate"])
                    self_employed  = st.selectbox("Self Employed", ["No", "Yes"])
                    property_area  = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

                st.subheader("Financial Information")
                f1, f2 = st.columns(2)
                with f1:
                    applicant_income    = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500)
                    coapplicant_income  = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=500)
                with f2:
                    loan_amount         = st.number_input("Loan Amount (₹ thousands)", min_value=1, value=150, step=10)
                    loan_term           = st.selectbox("Loan Term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)

                st.subheader("Credit Information")
                credit_history = st.selectbox("Credit History", [0, 1],
                                              format_func=lambda x: "No Credit History" if x == 0 else "Has Credit History")

                submitted = st.form_submit_button("🔍 Predict Loan Status", use_container_width=True)

        with col_result:
            if submitted:
                user_input = {
                    'Gender': gender, 'Married': married, 'Dependents': dependents,
                    'Education': education, 'Self_Employed': self_employed,
                    'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
                    'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_term,
                    'Credit_History': credit_history, 'Property_Area': property_area
                }
                prediction, probability = make_prediction(model, feature_names, pm, user_input)

                if prediction == 1:
                    st.markdown('<div class="approved">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="rejected">❌ LOAN REJECTED</div>', unsafe_allow_html=True)

                if probability is not None:
                    approval_prob  = probability[1]
                    rejection_prob = probability[0]
                    st.metric("Approval Probability", f"{approval_prob:.1%}")
                    st.metric("Model Confidence",     f"{max(probability):.1%}")

                    fig, ax = dark_fig((5, 3))
                    ax.bar(['Rejected', 'Approved'], [rejection_prob, approval_prob],
                           color=['#ef5350', '#43a047'])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Probability")
                    ax.set_title("Approval Probability Breakdown")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                total_inc = applicant_income + coapplicant_income
                dti = loan_amount / total_inc if total_inc > 0 else 0
                st.subheader("Application Summary")
                st.markdown(f"""
                | Field | Value |
                |---|---|
                | Total Income | ₹{total_inc:,} |
                | Loan Amount | ₹{loan_amount:,}K |
                | Debt-to-Income | {dti:.3f} |
                | Credit History | {"Yes" if credit_history == 1 else "No"} |
                | Property Area | {property_area} |
                | Education | {education} |
                """)

                # Contextual advice
                if prediction == 0:
                    reasons = []
                    if credit_history == 0:
                        reasons.append("No credit history is the #1 rejection driver — build credit first.")
                    if dti > 0.05:
                        reasons.append(f"Debt-to-income ratio of {dti:.3f} is high — consider a smaller loan.")
                    if total_inc < 3000:
                        reasons.append("Income is in the low bracket — adding a co-applicant may help.")
                    if reasons:
                        st.markdown("<div class='risk-box'>⚠️ <strong>Why rejected?</strong><br>" +
                                    "<br>".join(f"• {r}" for r in reasons) + "</div>",
                                    unsafe_allow_html=True)
            else:
                st.info("Complete the form and click **Predict Loan Status** to see results.")
                st.markdown("""
                **Model Info**
                - Algorithm: Best of SVM / Random Forest / XGBoost  
                - Accuracy: ~83%  
                - F1-Score: ~0.85  
                - Cross-Validation: 5-fold  

                **Top Predictive Features**
                1. Credit History  
                2. Total Income  
                3. Debt-to-Income Ratio  
                4. Property Area  
                5. Education  
                """)

    # Footer
    st.markdown("---")
    cols = st.columns(3)
    cols[0].caption("Stack: Python · Scikit-learn · XGBoost · Streamlit")
    cols[1].caption("Dataset: Kaggle — Loan Prediction Problem")
    cols[2].caption("Built by: Anurag Joshi")


if __name__ == "__main__":
    main()
