import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import multiprocessing

cores_num = multiprocessing.cpu_count()


sourcefilepath = "/data/project2_MTS_Regression/preprocess_bubble/"
sourcefullbubble = sourcefilepath+"all-2007-2023_extend_V_prep_total_gt_50_bubble_data.csv"
sourcebubblelist = sourcefilepath+"all-bubble_list_2007-2023_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_add_extend_dot_num.csv"
# 创建一个示例数据集

df_bubble = pd.read_csv(sourcefullbubble)
df_bubble_list = pd.read_csv(sourcebubblelist)
df_bubble['bubble_type'] = np.repeat(df_bubble_list['Yang_rating_second_time'].values,240,axis=0)
Index = ['Bx', 'By', 'Bz', 'B_total', 'Ni', 'Ne', 'plasma_beta', 'Pm', 'Pp','Ti', 'Te','B_inclination', 'T_N_ratio']
        
target_param = 'Vx'
# df_bubble['Bubble_Background_label'].isin([2])

# 使用前一个值填充空缺数据
df_bubble.fillna(method='pad', inplace=True)

# 提取bubble类型(1,2,3)为1, 且在bubble区域的数据(0-nonbubble,2-background,1-bubble)
df = df_bubble.loc[df_bubble['Bubble_Background_label'].isin([1]).values & df_bubble['bubble_type'].isin([1,2,3])]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[Index], df[target_param], test_size=0.2, random_state=42)

# 使用 Random Forest Regressor 进行训练
rf_model = RandomForestRegressor( n_estimators=1000,
                                 max_depth=16,
                                 n_jobs=cores_num)
rf_model.fit(X_train, y_train
        #      eval_metric='rmse',
        # early_stopping_rounds=5,
        # eval_set=[(X_train, y_train), (X_test, y_test)]
    )

joblib.dump(rf_model,'random_forest_model.joblib')
# 在测试集上进行预测
rf_model = joblib.load('random_forest_model.joblib')
y_pred = rf_model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# 计算 SHAP 值
explainer = shap.GPUTreeExplainer(rf_model)
shap_analysis = explainer(X_train)

df_SHAP = pd.DataFrame(
    data=shap_analysis.values,
    columns=Index
)

df_data = pd.DataFrame(
    data=shap_analysis.data,
    columns=Index
)

df_SHAP.to_csv(sourcefilepath+'/SHAP_Values.csv', index=None)
df_data.to_csv(
    sourcefilepath+'./Bubble_Parameters_Corresponding_to_SHAP_Values.csv',
    index=None
)

plt.rcParams['font.family'] = 'Arial'

shap.plots.beeswarm(shap_analysis, max_display=20)

path = sourcefilepath+'/Figure_shap.png'
plt.xlabel('SHAP value (R$_E$)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=600)

# with open('shap_values,pkl','wb') as file:
#     pickle.dump(shap_analysis,file)
# with open('shap_values.pkl','rb') as file:
#     loaded_shap_values = pickle.load(file)
# Summary plot of feature importance
# shap.summary_plot(loaded_shap_values, X_test)
# plt.show()

print('42')
