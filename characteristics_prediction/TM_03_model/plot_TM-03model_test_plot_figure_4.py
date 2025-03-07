import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Folder path containing Excel files
folder_path = "/Users/fengxuedong/Desktop/MTS_feature_regression/TSmodel_result/"

all_data = []

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Check for Excel files
        file_path = os.path.join(folder_path, file_name)
        # Read the Excel file
        df = pd.read_csv(file_path)
        # Optional: Add a column to identify the source file
        df['Source_File'] = file_name
        # Append the dataframe to the list
        all_data.append(df)

# Concatenate all dataframes
merged_data = pd.concat(all_data, ignore_index=True)
# Save the merged dataframe to a new Excel file
# merged_data.to_csv(folder_path+"merged_data_del_483.csv")
print("Files have been merged and saved as 'merged_output.xlsx'.")

data = merged_data
# 添加类别列
data['dataset_type'] = data['Dataset'].str.split('_').str[0]
data['parameter_type'] = data['Dataset'].str.split('_').str[1]
data['variable_type'] = data['Dataset'].str.split('_').str[2]
data['background_type'] = data['Source_File'].str.split("using_B_").str[1].str.split(".csv").str[0]
data.to_excel(folder_path+"merged_output.xlsx", index=False)

outputfilepath = '/Users/fengxuedong/Desktop/MTS_feature_regression/'


df = pd.DataFrame(data)

# 过滤测试集数据
test_df = df[(df["dataset_type"] == "Test") & (~df["variable_type"].str.contains("Range"))].reset_index()
test_df = test_df.drop(columns=['index','Dataset'])
test_df['background_type'] = test_df['background_type'].replace({'with_y_background': 'using actual background observations as additional inputs',
                                                        'TS_model': 'using background values derived from the TM-03', 
                                                        'without_y_background': 'without using additional background values'})

print(test_df)
# 设置绘图风格
sns.set(style="whitegrid")

# 绘制点线图
fig = plt.figure(figsize=(12, 6))
sns.lineplot(
    data=test_df,
    x="parameter_type",
    y="MAPE",
    hue="background_type",
    style="background_type",
    markers=True,
    dashes=False,
    palette="Set2"
)

# 添加标题和标签
# plt.title("MAPE Analysis by Parameter and Variable Type", fontsize=14)
plt.xlabel("Targets", fontsize=12)
plt.ylabel("MAPE (%)", fontsize=12)
plt.legend(title="Background Type", loc="upper right")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
fig.savefig(outputfilepath+'/predicted_result/figure8.pdf',dpi=1200,format='pdf')

# 显示图形
plt.show()
