# https://blog.csdn.net/AIHUBEI/article/details/119045370
import sklearn
print(sklearn.__version__)


from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features = 10, n_informative = 5, n_targets = 2, random_state = 1)

print(X.shape, y.shape)

#%% 1.用于多输出回归的线性回归
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

#%% 2.用于多输出回归的K近邻算法
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

#%% 3.用于多输出回归的随机森林回归
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

#%% 4.通过交叉验证对多输出回归进行评估

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


#%% 4.包装器多输出回归算法
# 为了实现SVR算法用于多输出回归，可以采用如下两种方法：
# 1. 为每个输出创建一个单独的模型；
# 2. 或者创建一个线性模型序列，其中每个模型的输出取决于先前模型的输出；

#%% 4.1 1.为每个输出创建单独的模型
# 代码示例
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR

# 创建数据集
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)

# 定义模型
model = LinearSVR()

# 将创建的模型对象作为参数传入
wrapper = MultiOutputRegressor(model)

# 训练模型
wrapper.fit(X, y)

# 使用包装器模型进行预测
data_in = [[-2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]

yhat = wrapper.predict(data_in)

# 预测结果汇总展示, 基于MultiOutputRegressor分别为每个输出训练了单独的模型
print(yhat)
#%% 4.2 2.为每个输出创建链式模型chained Models

# 代码示例，使用默认的输出顺序。基于multioutput regression 训练SVR

from sklearn.datasets import make_regression
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR

# 创建数据集
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)

# 定义模型
model  = LinearSVR()

wrapper = RegressorChain(model)

# 训练模型
wrapper.fit(X, y)
# 使用模型进行预测
data_in = [[-2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = wrapper.predict(data_in)

# 预测结果汇总输出
print(yhat)
