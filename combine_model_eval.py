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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from combine_model import CombinedAKIClassifier,CombinedAKIClassifier_v2
import numpy as np

# 1. 加载数据
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

best_params_xgb = {'max_depth': 6, 'learning_rate': 0.1406187747215624, 'n_estimators': 201, 'subsample': 0.9558079696044189, 'colsample_bytree': 0.5346372250120808, 'reg_alpha': 9.9402787182049, 'reg_lambda': 8.632921201766342}
best_params_stage_xgb = {'max_depth': 4, 'learning_rate': 0.08316606470999045, 'n_estimators': 216, 'subsample': 0.5261416788980062, 'colsample_bytree': 0.5233567963220875, 'reg_alpha': 5.776337504919956, 'reg_lambda': 9.62041035280805}
scaler = StandardScaler()
X = scaler.fit_transform(X_train)#[:,:-11]
X_test = scaler.transform(X_test)#[:,:-11]

model_name = 'xgb_model'
model = CombinedAKIClassifier_v2(xgb.XGBClassifier(**best_params_xgb, random_state=42),
xgb.XGBClassifier(**best_params_stage_xgb, random_state=42),
n_splits=5, prob_threshold=0.7)
#xgb.XGBClassifier(random_state=42)
#CombinedAKIClassifier(joblib.load(f'{model_name}.pkl'), joblib.load(f'stage_{model_name}.pkl'))

model.fit(X,y_train['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel(),y_train['aki_stage'].values.ravel())
# model.fit(X,y_train['aki_stage'].values.ravel())
y_pred = model.predict(X_test)

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of {model_name}: {accuracy:.4f}")

# 2. Classification Report
print(f"Classification Report for combine_{model_name}:\n{classification_report(y_test, y_pred)}")

# 3. Confusion Matrix
# If you have 4 classes, you can adjust the labels accordingly:
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No AKI','stage 1', 'stage 2', 'stage 3'],
            yticklabels=['No AKI','stage 1', 'stage 2', 'stage 3'])
plt.title(f"Confusion Matrix for {model_name}")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(f'combine_{model_name}_confusion_matrix.png', dpi=300)
plt.show()
plt.close()
# 4. ROC Curve and AUC (for multi-class)
# Binarize the output labels for multi-class ROC curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2,3])  # Adjust for 4 classes: classes=[0, 1, 2, 3]
y_pred_bin = model.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(4):  # Adjust the range for 4 classes: range(4)
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(6, 6))
for i in range(4):  # Adjust the range for 4 classes: range(4)
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for {model_name}')
plt.legend(loc='lower right')
plt.savefig(f'combine_{model_name}_roc_curve.png', dpi=300)
plt.show()
plt.close()
# 5. Precision-Recall Curve (for multi-class)
precision = dict()
recall = dict()
pr_auc = dict()

for i in range(4):  # Adjust the range for 4 classes: range(4)
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curve for each class
plt.figure(figsize=(6, 6))
for i in range(4):  # Adjust the range for 4 classes: range(4)
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AUC = {pr_auc[i]:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve for {model_name}')
plt.legend(loc='lower left')
plt.savefig(f'combine_{model_name}_precision_recall_curve.png', dpi=300)
plt.show()
plt.close()
# 6. ROC AUC Score
roc_auc_score_model = roc_auc_score(y_test_bin, y_pred_bin, average='macro', multi_class='ovr')
print(f"Macro-averaged ROC AUC Score of {model_name}: {roc_auc_score_model:.4f}")