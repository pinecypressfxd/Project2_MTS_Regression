# https://blog.csdn.net/AIHUBEI/article/details/119045370

# 检查sklearn版本
import sklearn
print(sklearn.__version__)

# sklearn.datasets中的多输出回归测试问题
from sklearn.datasets import make_regression
# 创建数据集
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2,random_state=1)

# 查看数据shape
print(X.shape, y.shape)


#%% 线性回归

# 代码
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# 创建数据集
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)

# 定义模型
model = LinearRegression()
# 训练模型
model.fit(X, y)
# 使用模型进行预测
data_in = [[-2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = model.predict(data_in)

# 预测结果的汇总
print(yhat)

#%%
# K近邻算法可以用于分类，回归。其中，用于回归的时候，采用均值法，用于分类的时候，一般采用投票法；
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor

# 创建数据集
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)

# 定义模型
model = KNeighborsRegressor()
# 训练模型
model.fit(X, y)

# 使用模型进行预测
data_in = [[-2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = model.predict(data_in)

# 预测结果的汇总
print(yhat)

#%%
# 代码示例
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# 创建数据
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)

# 定义模型
model = RandomForestRegressor()
# 训练模型
model.fit(X, y)

# 使用模型进行预测
data_in = [[-2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = model.predict(data_in)

# 预测结果的汇总
print(yhat)


#%% 交叉验证

# 使用交叉验证，对多输出回归进行评估
# 使用10折交叉验证，且重复三次
# 使用MAE作为模型的评估指标

from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# 创建数据集
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)

# 定义模型
model = DecisionTreeRegressor()

# 模型评估
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

# 结果汇总,结果在两个输出变量之间报告错误，而不是分别为每个输出变量进行单独的错误评分
n_scores = absolute(n_scores)

print("result:%.3f (%.3f)" %(mean(n_scores), std(n_scores)))
