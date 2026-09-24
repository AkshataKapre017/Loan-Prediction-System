"""
backend/model_utils.py
Shared constants, data loading, feature engineering, and prediction helpers.
Used by both train_model.py and frontend/app.py.
"""

import os
import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Resolve project root (one level above backend/)
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, 'data')
MODEL_DIR   = os.path.join(ROOT_DIR, 'model')

MODEL_FILE   = os.path.join(MODEL_DIR, 'best_loan_prediction_model.joblib')
FEATURE_FILE = os.path.join(MODEL_DIR, 'feature_names.joblib')
MAPPING_FILE = os.path.join(MODEL_DIR, 'preprocessing_mappings.joblib')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PREPROCESSING_MAPPINGS = {
    'Married':       {'No': 0, 'Yes': 1},
    'Gender':        {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education':     {'Graduate': 1, 'Not Graduate': 0}
}

# Embedded fallback dataset (Kaggle loan dataset format)
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
LP001291,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001293,Female,No,0,Graduate,No,3500,0,80,360,1,Semiurban,Y
LP001295,Male,Yes,1,Graduate,No,4833,2208,175,360,1,Urban,Y
LP001297,Male,Yes,2,Graduate,No,4833,0,125,360,1,Urban,Y
LP001300,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001301,Male,Yes,0,Graduate,No,6250,0,100,360,1,Urban,Y
LP001302,Male,Yes,2,Not Graduate,No,2666,0,66,360,0,Semiurban,N
LP001303,Male,Yes,0,Graduate,No,3833,0,100,360,1,Urban,Y
LP001309,Male,Yes,0,Not Graduate,No,3333,0,80,360,1,Rural,Y
LP001310,Male,Yes,0,Graduate,No,4166,0,100,360,1,Semiurban,Y
LP001311,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001314,Male,Yes,0,Graduate,No,5500,0,100,360,1,Urban,Y
LP001316,Female,No,0,Graduate,No,3500,0,80,360,1,Rural,Y
LP001317,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001318,Male,Yes,0,Graduate,No,3833,0,100,360,1,Semiurban,Y
LP001320,Male,Yes,0,Graduate,No,3250,2000,100,360,1,Urban,Y
LP001322,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001325,Male,Yes,2,Graduate,No,3750,0,100,360,1,Urban,Y
LP001326,Male,Yes,0,Graduate,No,4166,0,100,360,1,Semiurban,Y
LP001327,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001330,Male,Yes,0,Graduate,No,3500,0,80,360,1,Urban,Y
LP001332,Male,Yes,2,Graduate,No,4000,0,100,360,1,Urban,Y
LP001334,Male,Yes,0,Graduate,No,6250,0,100,360,1,Urban,Y
LP001338,Male,Yes,0,Graduate,No,5000,0,100,360,1,Semiurban,Y
LP001341,Female,No,0,Graduate,No,3500,0,80,360,1,Semiurban,Y
LP001344,Male,Yes,0,Graduate,No,5000,2500,150,360,1,Semiurban,Y
LP001348,Male,Yes,2,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001352,Male,Yes,0,Graduate,No,5833,0,100,360,1,Urban,Y
LP001355,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001358,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001360,Female,No,0,Graduate,No,3333,0,80,360,1,Semiurban,Y
LP001361,Male,Yes,1,Graduate,No,4166,2500,150,360,1,Urban,Y
LP001364,Male,Yes,0,Graduate,No,4583,0,100,360,1,Semiurban,Y
LP001366,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001367,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001370,Male,Yes,0,Graduate,No,3750,0,90,360,1,Urban,Y
LP001371,Male,Yes,2,Graduate,No,4583,1250,150,360,1,Urban,Y
LP001374,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001376,Male,Yes,0,Graduate,No,4166,0,100,360,1,Urban,Y
LP001378,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001382,Male,Yes,0,Graduate,No,5000,0,100,360,1,Semiurban,Y
LP001386,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001390,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001392,Female,No,0,Graduate,No,3500,0,80,360,1,Urban,Y
LP001394,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001398,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
LP001402,Male,Yes,0,Graduate,No,5000,0,100,360,1,Urban,Y
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data():
    """Load dataset from data/ folder, or fall back to embedded sample."""
    candidates = [
        os.path.join(DATA_DIR, 'loan_data.csv'),
        os.path.join(DATA_DIR, 'dataset.csv'),
        os.path.join(DATA_DIR, 'loan_data_set.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"  Loaded dataset from: {path}")
            return pd.read_csv(path)
    print("  No CSV found in data/ — using embedded sample dataset.")
    return pd.read_csv(io.StringIO(SAMPLE_DATA_CSV))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features_for_analytics(raw_df):
    """
    Return a human-readable copy for dashboard analytics.
    Keeps original categorical labels (no numeric encoding).
    """
    df = raw_df.copy()
    df = df.dropna(subset=['Loan_Status'])
    df['Dependents'] = df['Dependents'].replace('3+', 4)
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['DebtToIncomeRatio'] = np.where(
        df['TotalIncome'] > 0, df['LoanAmount'] / df['TotalIncome'], 0)
    df['LoanAmount_Category'] = pd.cut(
        df['LoanAmount'], bins=[0, 100, 200, 9999],
        labels=['Low (<=100K)', 'Medium (100-200K)', 'High (>200K)'])
    df['Income_Category'] = pd.cut(
        df['ApplicantIncome'], bins=[0, 3000, 6000, 99999],
        labels=['Low (<3K)', 'Medium (3-6K)', 'High (>6K)'])
    df['Approved'] = (df['Loan_Status'] == 'Y').astype(int)
    return df


def engineer_features_for_training(df):
    """
    Return a numerically encoded copy ready for ML training.
    Modifies in-place and returns X, Y.
    """
    df = df.copy()
    df = df.dropna()
    df.replace({'Loan_Status': {'N': 0, 'Y': 1}}, inplace=True)
    df = df.replace(to_replace='3+', value=4)
    df.replace({
        'Married':       {'No': 0, 'Yes': 1},
        'Gender':        {'Male': 1, 'Female': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Education':     {'Graduate': 1, 'Not Graduate': 0},
    }, inplace=True)
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['DebtToIncomeRatio'] = np.where(
        df['TotalIncome'] > 0, df['LoanAmount'] / df['TotalIncome'], 0)
    df['LoanAmount_Category'] = pd.cut(
        df['LoanAmount'], bins=[0, 100, 200, 9999], labels=[0, 1, 2])
    df['Income_Category'] = pd.cut(
        df['ApplicantIncome'], bins=[0, 3000, 6000, 99999], labels=[0, 1, 2])

    from sklearn.preprocessing import LabelEncoder
    X = df.drop(columns=['Loan_ID', 'Loan_Status'], errors='ignore')
    Y = df['Loan_Status']
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load the saved best model and metadata from model/ directory."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found at {MODEL_FILE}. Run train_model.py first.")
    model         = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURE_FILE)
    return model, feature_names, PREPROCESSING_MAPPINGS


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, feature_names, pm, user_input):
    """
    Run prediction for a single applicant dict.
    Returns (prediction: int, probability: array or None)
    """
    gender         = pm['Gender'][user_input['Gender']]
    married        = pm['Married'][user_input['Married']]
    education      = pm['Education'][user_input['Education']]
    self_employed  = pm['Self_Employed'][user_input['Self_Employed']]
    property_area  = pm['Property_Area'][user_input['Property_Area']]
    total_income   = user_input['ApplicantIncome'] + user_input['CoapplicantIncome']
    dti            = (user_input['LoanAmount'] / total_income) if total_income > 0 else 0
    loan_amt_cat   = 0 if user_input['LoanAmount'] <= 100 else 1 if user_input['LoanAmount'] <= 200 else 2
    income_cat     = 0 if user_input['ApplicantIncome'] <= 3000 else 1 if user_input['ApplicantIncome'] <= 6000 else 2

    feat = np.array([[
        gender, married, user_input['Dependents'], education, self_employed,
        user_input['ApplicantIncome'], user_input['CoapplicantIncome'],
        user_input['LoanAmount'], user_input['Loan_Amount_Term'],
        user_input['Credit_History'], property_area,
        total_income, dti, loan_amt_cat, income_cat
    ]])

    prediction  = model.predict(feat)[0]
    probability = model.predict_proba(feat)[0] if hasattr(model, 'predict_proba') else None
    return int(prediction), probability
