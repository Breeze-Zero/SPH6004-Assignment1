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

selected_features = ['gender', 'admission_age', 'race', 'heart_rate_max', 'sbp_min',
       'sbp_max', 'mbp_min', 'resp_rate_min', 'resp_rate_max',
       'temperature_vital_mean', 'spo2_min', 'glucose_vital_mean',
       'lactate_min', 'lactate_max', 'po2_min', 'po2_max', 'pco2_min',
       'baseexcess_min', 'baseexcess_max', 'totalco2_max',
       'hemoglobin_lab_min', 'hemoglobin_lab_max', 'platelets_min', 'wbc_min',
       'aniongap_min', 'aniongap_max', 'bun_min', 'bun_max', 'calcium_lab_min',
       'chloride_lab_min', 'creatinine_min', 'glucose_lab_min',
       'sodium_lab_min', 'potassium_lab_min', 'abs_basophils_min',
       'abs_basophils_max', 'abs_eosinophils_min', 'abs_lymphocytes_max',
       'abs_neutrophils_min', 'inr_max', 'ptt_min', 'gcs_verbal', 'gcs_unable',
       'weight_admit', 'scr_delta', 'scr_ratio', 'hypotension_flag',
       'bun_scr_ratio', 'interaction_hypotension_bun_scr', 'fever_flag',
       'interaction_fever_wbc']

# 1. 加载数据
X_train = pd.read_csv('X_train.csv').loc[:, selected_features]
X_test = pd.read_csv('X_test.csv').loc[:, selected_features]
y_train = pd.read_csv('y_train.csv')#['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()
y_test = pd.read_csv('y_test.csv')#['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()

mask = y_train['aki_stage'] != 0
X_train = X_train.loc[mask]
y_train = y_train.loc[mask]
y_train = y_train-1
y_train = y_train.values.ravel()

mask = y_test['aki_stage'] != 0
X_test = X_test.loc[mask]
y_test = y_test.loc[mask]
y_test = y_test-1
y_test = y_test.values.ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_params_nb = {'var_smoothing': 5.659522636392156e-07}
best_params_rf = {'n_estimators': 288, 'max_depth': 20, 'min_samples_split': 20, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'bootstrap': False}
best_params_svc = {'C': 4.023865627746826, 'gamma': 0.0023923757583377615, 'kernel': 'rbf'}
best_params_lr = {'C': 0.022293799684895, 'penalty': 'l2', 'solver': 'newton-cg'}
best_params_knn = {'n_neighbors': 30, 'weights': 'distance', 'algorithm': 'ball_tree'}
best_params_mlp = {'hidden_layer_sizes': (150,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0006314213159453925, 'learning_rate_init': 0.000897575706455529, 'early_stopping': True}
best_params_ada = {'n_estimators': 295, 'learning_rate': 0.21776599354601794}
best_params_xgb = {'max_depth': 4, 'learning_rate': 0.08316606470999045, 'n_estimators': 216, 'subsample': 0.5261416788980062, 'colsample_bytree': 0.5233567963220875, 'reg_alpha': 5.776337504919956, 'reg_lambda': 9.62041035280805}
best_params_gbdt = {'n_estimators': 235, 'learning_rate': 0.013584363770650346, 'max_depth': 9, 'subsample': 0.7296297311127056, 'max_features': 'sqrt'}

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
       cross_val_scores = cross_val_score(pipeline, X, y_train, cv=5, scoring='roc_auc_ovr',n_jobs=-1)
       print(f"{model_name} cross_val roc_auc: {cross_val_scores.mean():.4f}")
       # 训练模型
       pipeline.fit(X, y_train)
       joblib.dump(pipeline, f'stage_{model_name}.pkl')
       y_pred = pipeline.predict(X_test)
       print(classification_report(y_test, y_pred))