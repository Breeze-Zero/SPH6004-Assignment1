import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier,VotingClassifier
import joblib
from utils import remove_outliers_zscore,create_aki_features
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.pipeline import Pipeline
import itertools
from joblib import Parallel, delayed
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    joblib.dump(model, 'stacking_model.pkl')
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

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

# 1. 加载数据
X_train = pd.read_csv('X_train.csv').loc[:, selected_features]
X_test = pd.read_csv('X_test.csv').loc[:, selected_features]
y_train = pd.read_csv('y_train.csv')['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()
y_test = pd.read_csv('y_test.csv')['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()



# X_train = create_aki_features(X_train)
# X_test = create_aki_features(X_test)

# X_train["stage"] = X_train.apply(calculate_aki_stage, axis=1)
# X_test["stage"] = X_test.apply(calculate_aki_stage, axis=1)
# X_train,y_train = remove_outliers_zscore(X_train,y_train,selected_features, threshold=5)
# X_test,y_test = remove_outliers_zscore(X_test,y_test,selected_features)
# 3. 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


best_params_rf = {'n_estimators': 146, 'max_depth': 26, 'min_samples_split': 12, 'min_samples_leaf': 2, 'max_features': 'log2', 'bootstrap': False}
best_params_svc = {'C': 68.10438395969796, 'gamma': 0.0046650970854461545, 'kernel': 'rbf'}
best_params_lr = {'C': 60.801061547700456, 'penalty': 'l2', 'solver': 'lbfgs'}
best_params_mlp = {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'alpha': 5.599647344838225e-05, 'learning_rate_init': 0.012301043983120692, 'early_stopping': True}
best_params_ada = {'n_estimators': 248, 'learning_rate': 0.4442056458984761}
best_params_xgb = {'max_depth': 6, 'learning_rate': 0.1406187747215624, 'n_estimators': 201, 'subsample': 0.9558079696044189, 'colsample_bytree': 0.5346372250120808, 'reg_alpha': 9.9402787182049, 'reg_lambda': 8.632921201766342}
best_params_gbdt = {'n_estimators': 155, 'learning_rate': 0.02739221512374214, 'max_depth': 7, 'subsample': 0.8575194266893158, 'max_features': None}

ada_model = AdaBoostClassifier(**best_params_ada, random_state=42)
rf_model = RandomForestClassifier(**best_params_rf,random_state=42)
svc_model = SVC(**best_params_svc, probability=True, random_state=42)
lr_model = LogisticRegression(**best_params_lr,max_iter=1000)
mlp_model = MLPClassifier(**best_params_mlp, random_state=42, max_iter=1000)
xgb_model = xgb.XGBClassifier(**best_params_xgb, random_state=42)
gbdt_model = GradientBoostingClassifier(**best_params_gbdt, random_state=42)

base_models = [
    ('rf', rf_model),
    ('svc', svc_model),
    ('lr', lr_model),
    ('mlp', mlp_model),
    ('xgb', xgb_model),
    ('gbdt', gbdt_model),
    ('ada',ada_model)
]

final_layer = VotingClassifier(
    estimators=base_models,
    voting='soft',
    verbose=True,
    n_jobs=-1
)


stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=final_layer,
    cv=5, stack_method='auto', n_jobs=-1, passthrough=True,verbose=10
    )
# smote = SMOTE(sampling_strategy='minority', random_state=42)
# pipeline = Pipeline([('smote', smote), ('classifier', stack_model)])
#stack_model = joblib.load('stacking_model.pkl')
print("\nStacking model Evaluation:")
evaluate_model(stack_model, X_train, y_train, X_test, y_test)



# 生成所有可能的基模型组合（至少2个模型）
all_combinations = []
for r in range(2, len(base_models) + 1):
    all_combinations.extend(itertools.combinations(base_models, r))

# 定义评估函数
def evaluate_combination(combo):
    current_estimators = list(combo)
    model = VotingClassifier(
        estimators=current_estimators,
        voting='soft',
        verbose=False,
        n_jobs=1  # 防止嵌套并行
    )
    # 使用5折交叉验证，可根据需要调整cv和scoring参数
    scores = cross_val_score(model, X_train, y_train, cv=5, 
                            scoring='accuracy', n_jobs=1)
    return {
        'estimators': current_estimators,
        'score': scores.mean(),
        'names': [name for name, _ in current_estimators]
    }

# 并行评估所有组合
results = Parallel(n_jobs=-1)(
    delayed(evaluate_combination)(combo) for combo in all_combinations
)

# 找到最佳组合
best_result = max(results, key=lambda x: x['score'])

print(f"\n最佳组合: {best_result['names']}")
print(f"交叉验证平均准确率: {best_result['score']:.4f}")
