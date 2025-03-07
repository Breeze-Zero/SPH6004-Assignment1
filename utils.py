import numpy as np
import matplotlib.pyplot as plt
def plot_decision_boundary(model, X, y, title, cmap='rainbow'):
    # 生成网格数据用于绘制决策边界
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 预测网格数据类别
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(grid)[:, 1]
    else:
        Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    # 绘制填充等高线，颜色鲜艳
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=60, cmap=cmap)
    plt.title(title, fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.grid(True)
    plt.show()

# # 绘制XGBoost决策边界图
# plot_decision_boundary(xgb_model, X_train, y_train, title='XGBoost Decision Boundary Plot', cmap='plasma')

import pandas as pd
from scipy import stats
def remove_outliers_zscore(df,label_df, columns, threshold=3):
    """
    使用Z-Score方法剔除异常值所在行
    参数：
        threshold: Z分数阈值（默认±3）
    """
    z_scores = np.abs(stats.zscore(df[columns]))
    filter_mask = (z_scores < threshold).all(axis=1)
    return df[filter_mask].reset_index(drop=True),label_df[filter_mask]#.reset_index(drop=True)

def remove_outliers_iqr(df,label_df, columns, threshold=3):
    """
    使用IQR方法剔除指定列的异常值所在行
    参数：
        df: DataFrame
        columns: 需要处理的列名列表
        threshold: IQR倍数阈值（默认1.5）
    返回：
        过滤后的DataFrame
    """
    cleaned_df = df.copy()
    cleaned_label_df = label_df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # 保留在正常范围内的数据
        mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
        cleaned_df = cleaned_df[mask]
        cleaned_label_df = cleaned_label_df[mask]
    return cleaned_df.reset_index(drop=True),cleaned_label_df.reset_index(drop=True)


def calculate_egfr(row):
    scr = row['creatinine_min']  # 假设使用稳定状态下的肌酐最小值
    age = row['admission_age']
    is_male = 1 if row['gender'] == 1 else 0
    is_black = 1 if row['race'] == 1 else 0

    # CKD-EPI公式参数
    kappa = 0.9 if is_male else 0.7
    alpha = -0.411 if is_male else -0.329
    gender_coeff = 1.018 if not is_male else 1
    race_coeff = 1.159 if is_black else 1

    term1 = min(scr / kappa, 1) ** alpha
    term2 = max(scr / kappa, 1) ** (-1.209)
    term3 = 0.993 ** age

    egfr = 141 * term1 * term2 * term3 * gender_coeff * race_coeff
    return egfr

def calculate_hyperkalemia(row):
    potassium = row['potassium_lab_max']  
    if potassium>5.5:
        return 1
    else:
        return 0

def calculate_bum_scr(row):
    bun = (row['bun_min']+row['bun_max'])/2  
    scr = (row['creatinine_min']+row['creatinine_max'])/2 
    res = bun/scr
    if 10<=res<=20:
        return 0
    else:
        return 1

def calculate_aki_stage(row):
    """
    根据KDIGO标准判断AKI分期（仅肌酐指标）
    规则：
      - 3期: 当前值≥基线3倍 或 ≥4.0 mg/dl
      - 2期: 当前值≥基线2倍且<3倍
      - 1期: 当前值≥基线1.5倍 或 ↑≥0.3 mg/dl
      - 0期: 未达上述标准
    """
    baseline = row["creatinine_min"]
    current = row["creatinine_max"]
    
    # 处理无效数据
    if pd.isnull(baseline) or pd.isnull(current) or baseline <= 0:
        return -1
    
    # # # 优先检查绝对值阈值
    # if current >= 4.0:
    #     return 3
    
    # 计算相对变化
    ratio = current / baseline
    delta = current - baseline
    
    # 分期判断
    if ratio >= 3.0:
        return 3
    elif ratio >= 2.0:
        return 2
    elif ratio >= 1.5 or delta >= 0.3:
        return 1
    else:
        return 0

def create_aki_features(df):
    # 1. SCr change
    df['scr_delta'] = df['creatinine_max'] - df['creatinine_min']
    df['scr_ratio'] = df['creatinine_max'] / df['creatinine_min']
    
    # 2. haemodynamics
    df['hypotension_flag'] = ((df['sbp_min'] < 90) | (df['mbp_min'] < 65)).astype(int)
    df['shock_index'] = df['heart_rate_mean'] / df['sbp_mean']
    
    # 3. Metabolism and perfusion
    df['high_lactate_flag'] = (df['lactate_max'] > 2).astype(int)
    df['bun_scr_ratio'] = df['bun_max'] / df['creatinine_max']
    
    # 4. Inflammation and clotting
    df['siri_score'] = (df['wbc_max'] * df['abs_neutrophils_max']) / (df['abs_lymphocytes_min'] + 1e-6)
    df["egfr"] = df.apply(calculate_egfr, axis=1)

    # 5. Prerenal AKI interaction: hypotension × BUN/SCr ratio
    df['interaction_hypotension_bun_scr'] = df['hypotension_flag'] * df['bun_scr_ratio']
    
    # 6. Infectious AKI interaction: fever × increased WBC
    df['fever_flag'] = ((df['temperature_vital_max'] > 38) | (df['temperature_vital_min'] < 36)).astype(int)
    df['interaction_fever_wbc'] = df['fever_flag'] * (df['wbc_max'] > 12)
    
    return df