# Importing the Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True

print(" Starting Enhanced Loan Prediction Project")
print("="*50)

# ============================================================================
# DATA COLLECTION AND PROCESSING
# ============================================================================

print(" Loading and Processing Data...")

# Loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv('/content/dataset.csv')
print(f"Dataset shape: {loan_dataset.shape}")

# Display basic info
print("\nFirst 5 rows:")
print(loan_dataset.head())

# Check missing values
print(f"\nMissing values:\n{loan_dataset.isnull().sum()}")

# Handle missing values (using your original approach)
print(f"Rows before dropping missing values: {len(loan_dataset)}")
loan_dataset = loan_dataset.dropna()
print(f"Rows after dropping missing values: {len(loan_dataset)}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n Feature Engineering...")

# 1. Create TotalIncome feature
loan_dataset['TotalIncome'] = loan_dataset['ApplicantIncome'] + loan_dataset['CoapplicantIncome']

# 2. Create DebtToIncomeRatio feature
loan_dataset['DebtToIncomeRatio'] = loan_dataset['LoanAmount'] / loan_dataset['TotalIncome']
loan_dataset['DebtToIncomeRatio'] = loan_dataset['DebtToIncomeRatio'].fillna(0)

# 3. Bin LoanAmount into categories
loan_dataset['LoanAmount_Category'] = pd.cut(loan_dataset['LoanAmount'],
                                           bins=[0, 100, 200, float('inf')],
                                           labels=['Low', 'Medium', 'High'])

# 4. Bin ApplicantIncome into categories
loan_dataset['Income_Category'] = pd.cut(loan_dataset['ApplicantIncome'],
                                        bins=[0, 3000, 6000, float('inf')],
                                        labels=['Low', 'Medium', 'High'])

print(" New Features Created:")
print("- TotalIncome")
print("- DebtToIncomeRatio")
print("- LoanAmount_Category")
print("- Income_Category")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

print("\n Data Preprocessing...")

# Label encoding (keeping your original approach)
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}}, inplace=True)

# Handle 3+ in Dependents
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# Convert categorical columns to numerical values
loan_dataset.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'LoanAmount_Category': {'Low': 0, 'Medium': 1, 'High': 2},
    'Income_Category': {'Low': 0, 'Medium': 1, 'High': 2}
}, inplace=True)

# Convert Dependents column to numeric
loan_dataset['Dependents'] = pd.to_numeric(loan_dataset['Dependents'])

# Convert LoanAmount_Category and Income_Category to numeric
loan_dataset['LoanAmount_Category'] = loan_dataset['LoanAmount_Category'].cat.codes
loan_dataset['Income_Category'] = loan_dataset['Income_Category'].cat.codes

# ============================================================================
# DATA VISUALIZATION
# ============================================================================

print("\n Creating Visualizations...")

plt.figure(figsize=(15, 10))

# Original plots
plt.subplot(2, 3, 1)
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
plt.title('Education vs Loan Status')

plt.subplot(2, 3, 2)
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
plt.title('Marital Status vs Loan Status')

# New feature plots
plt.subplot(2, 3, 3)
sns.boxplot(x='Loan_Status', y='TotalIncome', data=loan_dataset)
plt.title('Total Income vs Loan Status')

plt.subplot(2, 3, 4)
sns.boxplot(x='Loan_Status', y='DebtToIncomeRatio', data=loan_dataset)
plt.title('Debt to Income Ratio vs Loan Status')

plt.subplot(2, 3, 5)
sns.countplot(x='LoanAmount_Category', hue='Loan_Status', data=loan_dataset)
plt.title('Loan Amount Category vs Loan Status')

plt.subplot(2, 3, 6)
sns.countplot(x='Income_Category', hue='Loan_Status', data=loan_dataset)
plt.title('Income Category vs Loan Status')

plt.tight_layout()
plt.show()

# ============================================================================
# PREPARE DATA FOR TRAINING
# ============================================================================

print("\n Preparing Data for Training...")

# Separating features and target
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

print(f"Feature shape: {X.shape}")
print(f"Target shape: {Y.shape}")

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
# Fix: Encode categorical/object columns
from sklearn.preprocessing import LabelEncoder
for col in X_train.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# ============================================================================
# MODEL TRAINING & COMPARISON
# ============================================================================

print("\n Training Multiple Models...")

# Initialize models
models = {
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    'SVM': svm.SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),

}

# Store results
results = {}

print("\nTraining Models:")
print("-" * 50)

for name, model in models.items():
    print(f"Training {name}...")

    # Train model
    model.fit(X_train, Y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)
    f1 = f1_score(Y_test, test_pred)
    roc_auc = roc_auc_score(Y_test, test_pred_proba) if test_pred_proba is not None else "N/A"

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()

    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_score': cv_mean,
        'test_predictions': test_pred,
        'test_probabilities': test_pred_proba
    }

    print(f" {name} completed!")

# ============================================================================
# MODEL COMPARISON TABLE 
# ============================================================================

print("\n MODEL COMPARISON RESULTS")
print("=" * 80)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[model]['train_accuracy'] for model in results.keys()],
    'Test Accuracy': [results[model]['test_accuracy'] for model in results.keys()],
    'F1 Score': [results[model]['f1_score'] for model in results.keys()],
    'ROC-AUC': [results[model]['roc_auc'] for model in results.keys()],
    'CV Score': [results[model]['cv_score'] for model in results.keys()]
})

# Format the dataframe for better display
for col in ['Train Accuracy', 'Test Accuracy', 'F1 Score', 'CV Score']:
    comparison_df[col] = comparison_df[col].round(4)

print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
best_model = results[best_model_name]['model']

print(f"\n Best Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")

# ============================================================================
# MODEL EVALUATION VISUALS 
# ============================================================================

print(f"\n Creating Evaluation Visuals for {best_model_name}...")

plt.figure(figsize=(15, 5))

# 1. Confusion Matrix
plt.subplot(1, 3, 1)
cm = confusion_matrix(Y_test, results[best_model_name]['test_predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. ROC Curve
plt.subplot(1, 3, 2)
if results[best_model_name]['test_probabilities'] is not None:
    fpr, tpr, _ = roc_curve(Y_test, results[best_model_name]['test_probabilities'])
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {results[best_model_name]["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {best_model_name}')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'ROC Curve not available\nfor this model',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title(f'ROC Curve - {best_model_name}')

# 3. Feature Importance (if available)
plt.subplot(1, 3, 3)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True)

    plt.barh(feature_importance['feature'][-10:], feature_importance['importance'][-10:])
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
elif hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(best_model.coef_[0])
    }).sort_values('importance', ascending=True)

    plt.barh(feature_importance['feature'][-10:], feature_importance['importance'][-10:])
    plt.title(f'Top 10 Feature Coefficients - {best_model_name}')
    plt.xlabel('Absolute Coefficient')
else:
    plt.text(0.5, 0.5, 'Feature importance\nnot available',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title(f'Feature Importance - {best_model_name}')

plt.tight_layout()
plt.show()

# ============================================================================
# CLASSIFICATION REPORT 
# ============================================================================

print(f"\n CLASSIFICATION REPORT - {best_model_name}")
print("=" * 50)
print(classification_report(Y_test, results[best_model_name]['test_predictions'],
                          target_names=['Rejected', 'Approved']))

# ============================================================================
# SAVE BEST MODEL
# ============================================================================

print(f"\n Saving Best Model ({best_model_name})...")

# Save the model
model_filename = 'best_loan_prediction_model.joblib'
joblib.dump(best_model, model_filename)

# Save feature names for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.joblib')

# Save preprocessing mappings for Streamlit app
preprocessing_mappings = {
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}
joblib.dump(preprocessing_mappings, 'preprocessing_mappings.joblib')

print(f" Model saved as: {model_filename}")
print(" Feature names saved as: feature_names.joblib")
print(" Preprocessing mappings saved as: preprocessing_mappings.joblib")

# ============================================================================
# PREDICTIVE SYSTEM DEMO 
# ============================================================================

print("\n PREDICTIVE SYSTEM DEMO")
print("=" * 40)

def make_prediction(applicant_income, coapplicant_income, loan_amount, loan_term,
                   credit_history, gender, married, dependents, education,
                   self_employed, property_area):
    """Enhanced prediction function with feature engineering"""

    # Create feature array with engineered features
    total_income = applicant_income + coapplicant_income
    debt_to_income = loan_amount / total_income if total_income > 0 else 0

    # Categorize loan amount
    if loan_amount <= 100:
        loan_amount_cat = 0  # Low
    elif loan_amount <= 200:
        loan_amount_cat = 1  # Medium
    else:
        loan_amount_cat = 2  # High

    # Categorize income
    if applicant_income <= 3000:
        income_cat = 0  # Low
    elif applicant_income <= 6000:
        income_cat = 1  # Medium
    else:
        income_cat = 2  # High

    # Create feature vector
    features = np.array([[gender, married, dependents, education, self_employed,
                         applicant_income, coapplicant_income, loan_amount, loan_term,
                         credit_history, property_area, total_income, debt_to_income,
                         loan_amount_cat, income_cat]])

    # Make prediction
    prediction = best_model.predict(features)[0]
    if hasattr(best_model, 'predict_proba'):
        probability = best_model.predict_proba(features)[0]
        return prediction, probability
    else:
        return prediction, None

# Demo prediction
print("Demo Prediction:")
prediction, probability = make_prediction(
    applicant_income=5000,
    coapplicant_income=2000,
    loan_amount=150,
    loan_term=360,
    credit_history=1,
    gender=1,  # Male
    married=1,  # Yes
    dependents=1,
    education=1,  # Graduate
    self_employed=0,  # No
    property_area=1  # Semiurban
)

result = "APPROVED " if prediction == 1 else "REJECTED"
print(f"Prediction: {result}")
if probability is not None:
    print(f"Probability of Approval: {probability[1]:.3f}")
    print(f"Confidence: {max(probability):.3f}")

print("Ready for Streamlit deployment")

# Show final dataset info
print(f"\nFinal Dataset Shape: {loan_dataset.shape}")
print(f"Features: {list(X.columns)}")
print(f"Best Model: {best_model_name} (Accuracy: {results[best_model_name]['test_accuracy']:.4f})")
