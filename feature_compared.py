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
       'RFE':['admission_age', 'race', 'heart_rate_min', 'heart_rate_max',
       'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean', 'dbp_max',
       'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min',
       'resp_rate_max', 'resp_rate_mean', 'temperature_vital_min',
       'temperature_vital_max', 'temperature_vital_mean', 'spo2_min',
       'spo2_max', 'spo2_mean', 'glucose_vital_min', 'glucose_vital_max',
       'glucose_vital_mean', 'lactate_min', 'lactate_max', 'ph_min', 'ph_max',
       'po2_min', 'pco2_max', 'baseexcess_min', 'baseexcess_max',
       'totalco2_min', 'totalco2_max', 'hematocrit_lab_min',
       'hematocrit_lab_max', 'hemoglobin_lab_min', 'hemoglobin_lab_max',
       'platelets_min', 'platelets_max', 'wbc_min', 'aniongap_min',
       'aniongap_max', 'bicarbonate_lab_min', 'bicarbonate_lab_max', 'bun_min',
       'bun_max', 'calcium_lab_min', 'calcium_lab_max', 'chloride_lab_min',
       'chloride_lab_max', 'creatinine_min', 'creatinine_max',
       'glucose_lab_min', 'glucose_lab_max', 'sodium_lab_min',
       'sodium_lab_max', 'potassium_lab_min', 'potassium_lab_max',
       'abs_basophils_max', 'abs_eosinophils_min', 'abs_lymphocytes_min',
       'abs_lymphocytes_max', 'abs_neutrophils_min', 'abs_neutrophils_max',
       'inr_min', 'inr_max', 'pt_min', 'ptt_min', 'ptt_max', 'gcs_min',
       'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'gcs_unable', 'weight_admit',
       'scr_delta', 'scr_ratio', 'hypotension_flag', 'shock_index',
       'high_lactate_flag', 'bun_scr_ratio', 'siri_score', 'egfr',
       'interaction_fever_wbc'],
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
       'all_features':X_train.columns,
       'all_features w/o Handing':X_train.columns[:-11],
       'lasso':['egfr', 'bun_max', 'chloride_lab_max', 'mbp_mean', 'bun_min',
        'dbp_mean', 'scr_ratio', 'weight_admit', 'gcs_verbal', 'hemoglobin_lab_min',
         'chloride_lab_min', 'sbp_mean', 'creatinine_min', 'hematocrit_lab_max',
          'resp_rate_mean', 'temperature_vital_max', 'sbp_max', 'pt_min',
           'calcium_lab_min', 'gcs_motor', 'lactate_min', 'bun_scr_ratio',
            'baseexcess_min', 'heart_rate_max', 'bicarbonate_lab_min', 'heart_rate_min',
             'ptt_max', 'platelets_min', 'gcs_eyes', 'hematocrit_lab_min', 'bicarbonate_lab_max',
              'hemoglobin_lab_max', 'inr_min', 'sbp_min', 'pco2_min', 'temperature_vital_mean',
               'gcs_min', 'baseexcess_max', 'admission_age', 'resp_rate_min', 'shock_index', 'hypotension_flag',
                'gcs_unable', 'totalco2_max', 'po2_min', 'calcium_lab_max', 'platelets_max', 'potassium_lab_max',
                 'mbp_min', 'spo2_max', 'high_lactate_flag', 'po2_max', 'sodium_lab_max', 'heart_rate_mean', 'aniongap_min',
                  'race', 'temperature_vital_min', 'gender', 'ptt_min', 'lactate_max', 'abs_neutrophils_max', 'sodium_lab_min', 'spo2_mean',
                   'mbp_max', 'abs_lymphocytes_max', 'interaction_hypotension_bun_scr', 'ph_max', 'dbp_max', 'glucose_vital_min', 'wbc_max',
                    'abs_basophils_max', 'wbc_min', 'abs_monocytes_min', 'siri_score', 'pt_max', 'glucose_vital_mean', 'inr_max',
                     'abs_lymphocytes_min', 'fever_flag', 'scr_delta', 'pco2_max', 'interaction_fever_wbc', 'spo2_min', 'resp_rate_max',
                      'abs_eosinophils_max', 'aniongap_max', 'glucose_lab_min', 'potassium_lab_min', 'abs_monocytes_max']
       }


scaler = StandardScaler()

for name,selected_feature in selected_features.items():
    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    X = X_train.loc[:, selected_feature]
    X = scaler.fit_transform(X)
    score = cross_val_score(model, X, y_train['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}), cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    print(f'{name} score:{score}')
