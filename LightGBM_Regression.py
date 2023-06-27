from sklearn.datasets import make_regression
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from lightgbm import LGBMRegressor
# define dataset
# test regresssion dataset
X,y = make_regression(n_samples=1000,n_features=20,n_informative=15,noise=0.1,random_state=7)
# summarize the dataset
print(X.shape, y.shape)
# define the model
model = LGBMRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats =3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1,error_score='raise')
# report performannce
print('MAE: %.3f (%.3f)' % (mean(n_scores),std(n_scores)))


# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [0.20543991,-0.97049844,-0.81403429,-0.23842689,-0.60704084,-0.48541492,0.53113006,2.01834338,-0.90745243,-1.85859731,-1.02334791,-0.6877744,0.60984819,-0.70630121,-1.29161497,1.32385441,1.42150747,1.26567231,2.56569098,-0.11154792]
yhat = model.predict([row])
print('Prediction: %d' % yhat[0])

