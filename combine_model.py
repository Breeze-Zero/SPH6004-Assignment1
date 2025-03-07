from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.base import clone
class CombinedAKIClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_classifier, stage_classifier):
        self.binary_classifier = binary_classifier  # 二分类模型（有无AKI）
        self.stage_classifier = stage_classifier    # 三分类模型（阶段0,1,2）

        self.binary_features = [1, 2, 4, 6, 7, 12, 13, 16, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35, 36, 39, 40, 41, 42, 43, 44, 45, 49, 51, 52, 55, 56, 57, 58, 61, 65, 68, 71, 73, 78, 79, 80, 81, 83, 85, 86, 87, 88, 91, 93, 94, 97]
        self.stage_features = [0, 1, 2, 4, 6, 7, 12, 15, 16, 20, 21, 26, 27, 28, 31, 32, 33, 35, 36, 38, 41, 42, 43, 45, 47, 48, 51, 52, 53, 55, 57, 59, 61, 63, 65, 66, 67, 70, 73, 76, 79, 83, 85, 86, 87, 88, 89, 92, 95, 96, 97]

    def fit(self, X, y=None):
        # 假设两个模型已预训练，此处无需操作
        return self
    
    def predict(self, X):
        # 使用二分类模型预测有无AKI
        binary_pred = self.binary_classifier.predict(X[:,self.binary_features])
        # 初始化结果全为0（无AKI）
        final_pred = np.zeros(X.shape[0], dtype=int)
        
        # 找出预测为有AKI的样本索引
        aki_indices = np.where(binary_pred == 1)[0]
        if len(aki_indices) > 0:
            print(len(aki_indices))
            X_aki = X[aki_indices]
            # 使用三分类模型预测阶段（0,1,2），并+1得到（1,2,3）
            stage_pred = self.stage_classifier.predict(X_aki[:,self.stage_features]) + 1
            final_pred[aki_indices] = stage_pred
        
        return final_pred

    # 可选：实现predict_proba（需模型支持概率预测）
    def predict_proba(self, X):
        binary_proba = self.binary_classifier.predict_proba(X[:,self.binary_features])
        proba = np.zeros((X.shape[0], 4))  # 4个类别的概率
        
        # 无AKI的概率（二分类的第0类）
        proba[:, 0] = binary_proba[:, 0]
        
        aki_indices = np.where(self.binary_classifier.predict(X[:,self.binary_features]) == 1)[0]
        if len(aki_indices) > 0:
            X_aki = X[aki_indices]
            stage_proba = self.stage_classifier.predict_proba(X_aki[:,self.stage_features])
            # 有AKI的概率乘以各阶段概率
            proba[aki_indices, 1:] = binary_proba[aki_indices, 1][:, np.newaxis] * stage_proba
        
        return proba



class CombinedAKIClassifier_v2(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_classifier, stage_classifier, 
                 n_splits=5, prob_threshold=0.7):
        self.binary_classifier = binary_classifier
        self.stage_classifier = stage_classifier
        self.n_splits = n_splits
        self.prob_threshold = prob_threshold  # 新增概率阈值参数
        self.binary_features = [1, 2, 4, 6, 7, 12, 13, 16, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35, 36, 39, 40, 41, 42, 43, 44, 45, 49, 51, 52, 55, 56, 57, 58, 61, 65, 68, 71, 73, 78, 79, 80, 81, 83, 85, 86, 87, 88, 91, 93, 94, 97]
        self.stage_features = [0, 1, 2, 4, 6, 7, 12, 15, 16, 20, 21, 26, 27, 28, 31, 32, 33, 35, 36, 38, 41, 42, 43, 45, 47, 48, 51, 52, 53, 55, 57, 59, 61, 63, 65, 66, 67, 70, 73, 76, 79, 83, 85, 86, 87, 88, 89, 92, 95, 96, 97]


    def fit(self, X, y_binary, y_stage):
        # 第一阶段：交叉验证获取二分类概率预测
        kf = KFold(n_splits=self.n_splits)
        binary_probas = np.zeros(X.shape[0])  # 存储概率值
        
        for train_idx, val_idx in kf.split(X,y_binary):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train_binary = y_binary[train_idx]
            
            # 克隆模型防止污染原始模型
            fold_model = clone(self.binary_classifier)
            fold_model.fit(X_train[:, self.binary_features], y_train_binary)
            
            # 获取验证集属于类别1的概率
            proba = fold_model.predict_proba(X_val[:, self.binary_features])[:, 1]
            binary_probas[val_idx] = proba
        
        # 第二阶段：用概率阈值筛选训练数据
        # 筛选条件：二分类概率 > 阈值 且 真实阶段标签非0
        selected = np.where(
            (binary_probas > self.prob_threshold) & 
            (y_stage != 0)
        )[0]
        
        print(f"Selected {len(selected)} samples for stage training.")
        
        # 处理样本不足的边界情况
        if len(selected) < 10:
            raise ValueError(
                f"仅筛选到{len(selected)}个样本，请降低prob_threshold或检查数据分布。"
                "建议阈值范围：0.3~0.8")
        
        X_stage = X[selected]
        y_stage_labels = y_stage[selected] - 1  # 转换为0,1,2
        
        # 第三阶段：全量训练最终模型
        self.binary_classifier.fit(X[:, self.binary_features], y_binary)
        self.stage_classifier.fit(X_stage[:, self.stage_features], y_stage_labels)
        
        return self

    def predict(self, X):
        # 预测时同样应用概率阈值
        binary_proba = self.binary_classifier.predict_proba(
            X[:, self.binary_features])[:, 1]
        final_pred = np.zeros(X.shape[0], dtype=int)
        
        # 筛选预测概率超过阈值的样本
        aki_indices = np.where(binary_proba > self.prob_threshold)[0]
        if len(aki_indices) > 0:
            X_aki = X[aki_indices]
            stage_pred = self.stage_classifier.predict(
                X_aki[:, self.stage_features]) + 1
            final_pred[aki_indices] = stage_pred
        
        return final_pred

    def predict_proba(self, X):
        binary_proba = self.binary_classifier.predict_proba(X[:,self.binary_features])
        proba = np.zeros((X.shape[0], 4))  # 4个类别的概率
        
        # 无AKI的概率（二分类的第0类）
        proba[:, 0] = binary_proba[:, 0]
        
        aki_indices = np.where(self.binary_classifier.predict(X[:,self.binary_features]) == 1)[0]
        if len(aki_indices) > 0:
            X_aki = X[aki_indices]
            stage_proba = self.stage_classifier.predict_proba(X_aki[:,self.stage_features])
            # 有AKI的概率乘以各阶段概率
            proba[aki_indices, 1:] = binary_proba[aki_indices, 1][:, np.newaxis] * stage_proba
        
        return proba