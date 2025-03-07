import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import random
import os
from utils import create_aki_features
# 创建映射函数
def categorize_race(race):
    race = str(race).upper()
    if 'BLACK' in race or 'AFRICAN AMERICAN' in race:
        return 'BLACK'
    elif 'WHITE' in race:
        return 'WHITE'
    elif 'ASIAN' in race or 'PACIFIC' in race:
        return 'ASIAN'
    else:
        return 'OTHER'


mice_imputer = IterativeImputer(max_iter=50, random_state=42)

if os.path.exists('data_imputed.csv'):
    df = pd.read_csv('data_imputed.csv')
else:
    df = pd.read_csv('sph6004_assignment1_data.csv')
    # df = df.drop(columns=['race'])
    df['race'] = df['race'].apply(categorize_race).map({'WHITE': 0, 'BLACK': 1, 'ASIAN': 2, 'OTHER': 3})
    df = df.drop(columns=['id'])
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})
    # df = df.dropna(axis=1, thresh=int(0.3 * len(df)))
    # df = df.dropna(axis=0, how='any')
    # drop NA
    missing_ratio = df.isna().mean()
    col_threshold = 0.5
    cols_to_drop = missing_ratio[missing_ratio > col_threshold].index
    df.drop(columns=cols_to_drop, inplace=True)
    row_threshold = 0.5
    df = df[df.isna().mean(axis=1) <= row_threshold]
    df = create_aki_features(df)
    label_columns = ['hospital_mortality','aki_stage']
    concat_df = df[label_columns]
    imputed_df = df.drop(columns=label_columns)
    # if concat_df['aki_stage'].isna().any():
    #     print("label 列存在缺失值 (NA)。")
    # else:
    #     print("label 列没有缺失值。")

    data_imputed = mice_imputer.fit_transform(imputed_df)
    df = pd.DataFrame(data_imputed, columns=imputed_df.columns)
    # df[label_columns] = concat_df
    concat_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, concat_df], axis=1)
    df.to_csv('data_imputed.csv', index=False)

if os.path.exists('X_train.csv'):
    pass
else:
    aki_df = df.drop(columns=['hospital_mortality'])
    # aki_df['aki_stage'] = aki_df['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1})
    X = aki_df.drop(columns=['aki_stage'])
    y = aki_df['aki_stage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
scaler = StandardScaler()
# 遗传算法筛选特征
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
from tqdm import tqdm 

# 特征选择的适应度函数
def evaluate(individual, X, y):
    selected_features = [index for index in range(len(individual)) if individual[index] == 1]
    
    if len(selected_features) == 0:
        return 0,  # 避免选择空的特征集
    
    X_selected = X[:, selected_features]
    
    # 使用交叉验证评估模型
    classifier = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X_selected, y, cv=cv, scoring='accuracy')
    
    return np.mean(scores),

# 定义目标和适应度函数
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    return [random.randint(0, 1) for _ in range(X_train.shape[1])]

# 初始化种群
toolbox = base.Toolbox()

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册遗传算法操作
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate, X=scaler.fit_transform(X_train), y=y_train['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel())

# 设置并行计算（如果需要）
def parallel_eval(individual):
    return evaluate(individual, X=scaler.fit_transform(X_train), y=y_train['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel())

# 开启进程池进行并行化
pool = Pool(processes=None)  # 默认为CPU核数
toolbox.register("map", pool.map)

# 遗传算法设置
pop = toolbox.population(n=X_train.shape[1])
hall_of_fame = tools.HallOfFame(5)

# 使用Stats对象收集统计信息
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# 在遗传算法中使用tqdm显示进度条
def update_progress_bar(gen, max_gen, result, stats, hall_of_fame):
    """ 用tqdm更新进度条 """
    gen_stats = stats.compile(result[0])
    print(f"Generation {gen}/{max_gen}, Best fitness: {gen_stats['max']:.4f}, Avg fitness: {gen_stats['avg']:.4f}")
    hall_of_fame.update(result[0])  # 更新最优个体
    return gen_stats

# 运行遗传算法并显示进度条
with tqdm(total=100, desc="Running GA", ncols=100) as pbar:
    for gen in range(100):
        # 运行一代的遗传算法
        result = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=1, 
                                      stats=stats, halloffame=hall_of_fame, verbose=True)
        update_progress_bar(gen + 1, 100, result, stats, hall_of_fame)
        pbar.update(1)  # 更新进度条

# 输出最优特征子集
best_individual = tools.selBest(pop, 1)[0]
print("Best individual (feature subset):", best_individual)

# 获取最优特征子集
selected_features = [index for index in range(len(best_individual)) if best_individual[index] == 1]
# X_best = X[:, selected_features]

# 将特征列索引转换为列名
selected_features = X_train.columns[selected_features]
print(selected_features)

selected_X = X_train.loc[:, selected_features]
selected_X = scaler.fit_transform(selected_X)
y = y_train.map({0: 0, 1: 1, 2:1, 3:1}).values.ravel()
test = X_test.loc[:, selected_features]
test = scaler.fit_transform(test)
# smote = SMOTE(random_state=42)
# selected_X, y = smote.fit_resample(selected_X, y_train.values.ravel())
model = RandomForestClassifier(random_state=42)
model.fit(selected_X, y)

# 在测试集上进行预测
y_pred = model.predict(test)
# 评估准确率
accuracy = accuracy_score(y_test['aki_stage'].map({0: 0, 1: 1, 2:1, 3:1}).values.ravel(), y_pred)
print(f"模型的准确率: {accuracy * 100:.2f}%")