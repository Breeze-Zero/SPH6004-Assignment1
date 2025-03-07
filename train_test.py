from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.pipeline import Pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

selected_features = ['admission_age', 'race', 'heart_rate_max', 'sbp_min', 'sbp_max',
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
       'high_lactate_flag', 'siri_score', 'egfr', 'interaction_fever_wbc']
X = pd.read_csv('X_train.csv').loc[:, selected_features]

y = pd.read_csv('y_train.csv')['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()

X_test = pd.read_csv('X_test.csv').loc[:, selected_features]
y_test = pd.read_csv('y_test.csv')['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

best_params_nb = {'var_smoothing': 9.732524675926293e-07}
best_params_rf = {'n_estimators': 146, 'max_depth': 26, 'min_samples_split': 12, 'min_samples_leaf': 2, 'max_features': 'log2', 'bootstrap': False}
best_params_svc = {'C': 68.10438395969796, 'gamma': 0.0046650970854461545, 'kernel': 'rbf'}
best_params_lr = {'C': 60.801061547700456, 'penalty': 'l2', 'solver': 'lbfgs'}
best_params_knn = {'n_neighbors': 30, 'weights': 'distance', 'algorithm': 'auto'}
best_params_mlp = {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'alpha': 5.599647344838225e-05, 'learning_rate_init': 0.012301043983120692, 'early_stopping': True}
best_params_ada = {'n_estimators': 248, 'learning_rate': 0.4442056458984761}
best_params_xgb = {'max_depth': 6, 'learning_rate': 0.1406187747215624, 'n_estimators': 201, 'subsample': 0.9558079696044189, 'colsample_bytree': 0.5346372250120808, 'reg_alpha': 9.9402787182049, 'reg_lambda': 8.632921201766342}
best_params_gbdt = {'n_estimators': 155, 'learning_rate': 0.02739221512374214, 'max_depth': 7, 'subsample': 0.8575194266893158, 'max_features': None}

model_dict = {'nb_model':GaussianNB(**best_params_nb),
'knn_model':KNeighborsClassifier(**best_params_knn),
'ada_model' : AdaBoostClassifier(**best_params_ada, random_state=42),
'rf_model' : RandomForestClassifier(**best_params_rf,random_state=42),
'svc_model' : SVC(**best_params_svc, probability=True, random_state=42),
'lr_model' : LogisticRegression(**best_params_lr,max_iter=1000),
'mlp_model' : MLPClassifier(**best_params_mlp, random_state=42, max_iter=1000),
'xgb_model' : xgb.XGBClassifier(**best_params_xgb, random_state=42),
'gbdt_model' : GradientBoostingClassifier(**best_params_gbdt, random_state=42)}
# 定义ADASYN过采样器
adasyn = ADASYN(sampling_strategy='minority', random_state=42)
smote = SMOTE(sampling_strategy='minority', random_state=42)
for model_name,model in model_dict.items():
       pipeline = model#Pipeline([('smote', smote), ('classifier', model)])
       # 使用交叉验证评估模型性能
       cross_val_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc',n_jobs=-1)
       print(f"{model_name} cross_val roc_auc: {cross_val_scores.mean():.4f}")
       # 训练模型
       pipeline.fit(X, y)
       joblib.dump(pipeline, f'{model_name}.pkl')
       y_pred = pipeline.predict(X_test)
       print(classification_report(y_test, y_pred))