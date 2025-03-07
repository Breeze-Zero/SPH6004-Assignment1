import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import optuna
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


# 1. 加载数据
X_train = pd.read_csv('X_train.csv').loc[:, selected_features]
X_test = pd.read_csv('X_test.csv').loc[:, selected_features]
y_train = pd.read_csv('y_train.csv')['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()
y_test = pd.read_csv('y_test.csv')['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()


# 3. 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. 定义目标函数用于Optuna
def objective_rf(trial):
    # rf param
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    model = RandomForestClassifier(**param_grid, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

def objective_nb(trial):
    # nb param
    param_grid = {
        'var_smoothing': trial.suggest_loguniform('var_smoothing', 1e-12, 1e-6)
    }
    model = GaussianNB(**param_grid)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

def objective_svc(trial):
    # svc param
    param_grid = {
        'C': trial.suggest_loguniform('C', 1e-5, 100),
        'gamma': trial.suggest_loguniform('gamma', 1e-5, 100),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf'])
    }
    model = SVC(**param_grid, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

def objective_lr(trial):
    # LR param
    param_grid = {
        'C': trial.suggest_loguniform('C', 1e-5, 100),
        'penalty': trial.suggest_categorical('penalty', ['l2']),#'l1',
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga',"lbfgs",'newton-cholesky','newton-cg'])#multi_class="multinomial", solver="lbfgs"
    }
    '''
    The choice of the algorithm depends on the penalty chosen and on (multinomial) multiclass support:
    solver penalty multinomial multiclass
    ‘lbfgs’ ‘l2’, None yes
    ‘liblinear’ ‘l1’, ‘l2’ no
    ‘newton-cg’ ‘l2’, None yes
    ‘newton-cholesky’ ‘l2’, None no
    ‘sag’ ‘l2’, None yes 
    ‘saga’ ‘elasticnet’, ‘l1’, ‘l2’, None yes
    '''
    model = LogisticRegression(**param_grid, random_state=42,max_iter=1000)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

def objective_knn(trial):
    # Knn param
    param_grid = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
    }
    model = KNeighborsClassifier(**param_grid,n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

def objective_mlp(trial):
    # MLP param
    param_grid = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (150,)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-2),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1),
        'early_stopping': trial.suggest_categorical('early_stopping', [True])
    }
    model = MLPClassifier(**param_grid, random_state=42, max_iter=1000)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

def objective_adaboost(trial):
    # AdaBoost超参数
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1)
    }
    model = AdaBoostClassifier(**param_grid, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

def objective_xgb(trial):
    # XGBoost param
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  # 添加正则化
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }
    model = xgb.XGBClassifier(**param_grid, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

## TODO ADD LGBT


def objective_gbdt(trial):
    # GBDT param
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    model = GradientBoostingClassifier(**param_grid, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()
    return score

# 5. 使用Optuna对每个模型进行超参数调优
def optimize_model(objective_function, n_trials=50,n_jobs=-1):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_function, n_trials=n_trials, n_jobs=n_jobs)#
    return study.best_params, study.best_value

# 选择优化的模型
best_params_nb, best_score_nb = optimize_model(objective_nb)
best_params_rf, best_score_rf = optimize_model(objective_rf)
best_params_svc, best_score_svc = optimize_model(objective_svc)
best_params_lr, best_score_lr = optimize_model(objective_lr)
best_params_knn, best_score_knn = optimize_model(objective_knn,n_jobs=1)
best_params_mlp, best_score_mlp = optimize_model(objective_mlp)
best_params_adaboost, best_score_adaboost = optimize_model(objective_adaboost)
best_params_xgb, best_score_xgb = optimize_model(objective_xgb)
best_params_gbdt, best_score_gbdt = optimize_model(objective_gbdt)
# best_params_nb = {'var_smoothing': 9.732524675926293e-07}
# best_params_rf = {'n_estimators': 146, 'max_depth': 26, 'min_samples_split': 12, 'min_samples_leaf': 2, 'max_features': 'log2', 'bootstrap': False}
# best_params_svc = {'C': 68.10438395969796, 'gamma': 0.0046650970854461545, 'kernel': 'rbf'}
# best_params_lr = {'C': 60.801061547700456, 'penalty': 'l2', 'solver': 'lbfgs'}
# best_params_knn = {'n_neighbors': 30, 'weights': 'distance', 'algorithm': 'auto'}
# best_params_mlp = {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'alpha': 5.599647344838225e-05, 'learning_rate_init': 0.012301043983120692, 'early_stopping': True}
# best_params_adaboost = {'n_estimators': 248, 'learning_rate': 0.4442056458984761}
# best_params_xgb = {'max_depth': 6, 'learning_rate': 0.1406187747215624, 'n_estimators': 201, 'subsample': 0.9558079696044189, 'colsample_bytree': 0.5346372250120808, 'reg_alpha': 9.9402787182049, 'reg_lambda': 8.632921201766342}
# best_params_gbdt = {'n_estimators': 155, 'learning_rate': 0.02739221512374214, 'max_depth': 7, 'subsample': 0.8575194266893158, 'max_features': None}
# 6. 输出最佳超参数
print("Best GaussianNB Params:", best_params_nb)
print("Best Random Forest Params:", best_params_rf)
print("Best SVC Params:", best_params_svc)
print("Best Logistic Regression Params:", best_params_lr)
print("Best KNN Params:", best_params_knn)
print("Best MLP Params:", best_params_mlp)
print("Best AdaBoost Params:", best_params_adaboost)
print("Best XGBoost Params:", best_params_xgb)
print("Best GBDT Params:", best_params_gbdt)

# 7. 训练最终模型并评估
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 训练和评估最好的模型
nb_model = GaussianNB(**best_params_nb)
rf_model = RandomForestClassifier(**best_params_rf, random_state=42,n_jobs=-1)
svc_model = SVC(**best_params_svc, probability=True, random_state=42)
lr_model = LogisticRegression(**best_params_lr, random_state=42,max_iter=1000)
knn_model = KNeighborsClassifier(**best_params_knn,n_jobs=-1)
mlp_model = MLPClassifier(**best_params_mlp, random_state=42,max_iter=1000)
adaboost_model = AdaBoostClassifier(**best_params_adaboost, random_state=42)
xgb_model = xgb.XGBClassifier(**best_params_xgb, random_state=42,n_jobs=-1)
gbdt_model = GradientBoostingClassifier(**best_params_gbdt, random_state=42)

# 评估模型
print("\nGaussianNB Evaluation:")
evaluate_model(nb_model, X_train, y_train, X_test, y_test)

print("\nRandom Forest Evaluation:")
evaluate_model(rf_model, X_train, y_train, X_test, y_test)

print("\nSVC Evaluation:")
evaluate_model(svc_model, X_train, y_train, X_test, y_test)

print("\nLogistic Regression Evaluation:")
evaluate_model(lr_model, X_train, y_train, X_test, y_test)

print("\nKNN Evaluation:")
evaluate_model(knn_model, X_train, y_train, X_test, y_test)

print("\nMLP Evaluation:")
evaluate_model(mlp_model, X_train, y_train, X_test, y_test)

print("\nAdaBoost Evaluation:")
evaluate_model(adaboost_model, X_train, y_train, X_test, y_test)

print("\nXGBoost Evaluation:")
evaluate_model(xgb_model, X_train, y_train, X_test, y_test)

print("\nGBDT Evaluation:")
evaluate_model(gbdt_model, X_train, y_train, X_test, y_test)
