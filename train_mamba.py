import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from mambular.models import MambularClassifier


X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')



selected_features = {
       'Genetic':['admission_age', 'race', 'heart_rate_max', 'sbp_min', 'sbp_max',
       'mbp_min', 'mbp_max', 'resp_rate_max', 'temperature_vital_min',
       'temperature_vital_max', 'spo2_min', 'spo2_max', 'glucose_vital_max',
       'glucose_vital_mean', 'lactate_max', 'ph_min', 'po2_max',
       'baseexcess_min', 'baseexcess_max', 'hematocrit_lab_min',
       'hematocrit_lab_max', 'hemoglobin_lab_min', 'hemoglobin_lab_max',
       'platelets_min', 'platelets_max', 'wbc_min', 'bicarbonate_lab_min',
       'bun_min', 'bun_max', 'chloride_lab_min', 'chloride_lab_max',
       'creatinine_min', 'creatinine_max', 'sodium_lab_min',
       'abs_basophils_min', 'abs_eosinophils_max', 'abs_monocytes_min',
       'abs_neutrophils_min', 'pt_max', 'ptt_min', 'ptt_max', 'gcs_min',
       'gcs_verbal', 'gcs_unable', 'weight_admit', 'scr_delta', 'scr_ratio',
       'high_lactate_flag', 'siri_score', 'egfr', 'interaction_fever_wbc'],
       }


model = MambularClassifier(
    d_model=64,#64,
    n_layers=4,#4,
    numerical_preprocessing="ple",
    n_bins=50,
    d_conv=8,
    lr=1e-4
)

X = X_train.loc[:, selected_features['Genetic']]
y = y_train['aki_stage']#.map({0: 0, 1: 1, 2:1, 3:1})
model.fit(X, y, max_epochs=100,dataloader_kwargs={'num_workers':8})

print(model.evaluate(X_test.loc[:, selected_features['Genetic']], y_test['aki_stage']))