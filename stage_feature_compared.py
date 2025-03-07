import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

mask = y_train['aki_stage'] != 0
X_train = X_train.loc[mask]
y_train = y_train.loc[mask]
y_train = y_train-1

mask = y_test['aki_stage'] != 0
X_test = X_test.loc[mask]
y_test = y_test.loc[mask]
y_test = y_test-1

# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# scaler = StandardScaler()
# X = scaler.fit_transform(X_train)
# # 使用逻辑回归进行特征选择
# model = LogisticRegression(max_iter=1000)
# selector = RFE(model, n_features_to_select=10)
# param_grid = {
#     'n_features_to_select': [i for i in range(5,len(X_train.columns))]  # 设置候选的特征数量
# }
# grid_search = GridSearchCV(selector, param_grid,
#                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
#                            scoring='accuracy',
#                            n_jobs=-1)

# grid_search.fit(X, y_train['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel())

# # 输出最好的参数和最佳得分
# print(f"Best number of features: {grid_search.best_params_['n_features_to_select']}")
# best_selector = grid_search.best_estimator_

# selected_features = X_train.columns[best_selector.support_]
# print(selected_features)

# from sklearn.linear_model import Lasso,LassoCV
# scaler = StandardScaler()
# X = scaler.fit_transform(X_train)
# # 使用Lasso回归进行特征选择
# model = LassoCV(cv=5, random_state=42, n_jobs=-1)
# model.fit(X, y_train.values.ravel())
# lasso_coefficients = model.coef_

# # 输出最佳的 alpha 值
# print(f"Best alpha value: {model.alpha_}")
# # 创建DataFrame展示特征及其系数
# lasso_df = pd.DataFrame({'feature': X_train.columns, 'coefficient': np.abs(lasso_coefficients)})
# lasso_df = lasso_df[lasso_df['coefficient'] != 0].sort_values(by='coefficient', ascending=False)
# print(list(lasso_df['feature']))

selected_features = {
       'RFE':['gender', 'admission_age', 'heart_rate_min', 'heart_rate_max',
       'heart_rate_mean', 'sbp_max', 'dbp_mean', 'mbp_min', 'mbp_mean',
       'resp_rate_mean', 'temperature_vital_max', 'temperature_vital_mean',
       'spo2_max', 'lactate_min', 'ph_min', 'ph_max', 'po2_max', 'pco2_min',
       'baseexcess_min', 'baseexcess_max', 'totalco2_min',
       'hematocrit_lab_min', 'hematocrit_lab_max', 'hemoglobin_lab_min',
       'hemoglobin_lab_max', 'platelets_min', 'platelets_max', 'aniongap_min',
       'bun_min', 'bun_max', 'calcium_lab_min', 'calcium_lab_max',
       'chloride_lab_min', 'chloride_lab_max', 'abs_lymphocytes_min',
       'abs_lymphocytes_max', 'abs_neutrophils_min', 'abs_neutrophils_max',
       'inr_min', 'pt_min', 'ptt_max', 'gcs_min', 'gcs_motor', 'gcs_verbal',
       'gcs_eyes', 'gcs_unable', 'weight_admit', 'scr_ratio',
       'hypotension_flag', 'shock_index', 'bun_scr_ratio', 'egfr',
       'interaction_hypotension_bun_scr'],
       'Genetic':['gender', 'admission_age', 'race', 'heart_rate_max', 'sbp_min',
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
       'interaction_fever_wbc'],
       'all_features':X_train.columns,
       'all_features w/o Handing':X_train.columns[:-11],
       'lasso':['egfr', 'gcs_verbal', 'scr_ratio', 'weight_admit', 'ptt_max', 'chloride_lab_max', 'calcium_lab_min', 'heart_rate_max', 'resp_rate_mean', 'po2_max', 'bun_max', 'po2_min', 'mbp_min', 'aniongap_max', 'creatinine_max', 'aniongap_min',
        'mbp_max', 'hemoglobin_lab_max', 'pt_min', 'sbp_min', 'pt_max', 'fever_flag', 'lactate_max', 'gcs_min', 'dbp_min']
       }


scaler = StandardScaler()

for name,selected_feature in selected_features.items():
    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    X = X_train.loc[:, selected_feature]
    X = scaler.fit_transform(X)
    score = cross_val_score(model, X, y_train['aki_stage'], cv=StratifiedKFold(n_splits=5), scoring='roc_auc_ovr').mean()
    print(f'{name} score:{score}')
