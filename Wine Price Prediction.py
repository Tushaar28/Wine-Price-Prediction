import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

wine = pd.read_csv('https://storage.googleapis.com/dimensionless/Analytics/wine.csv')


wine.dtypes


wine.describe()

wine.head()

model1 = sm.ols(formula='Price ~ AGST',data=wine).fit() # Or model1 = sm.ols('Price ~ AGST',data=wine).fit()

plt.plot(wine['AGST'],wine['Price'],'ro')
plt.plot(wine['AGST'],model1.fittedvalues,'b')
plt.xlabel('AGST')
plt.ylabel('Price')
plt.title('Model-1 (Price ~ AGST)')
plt.show()


model1.summary()


# In[248]:


model1.params


# In[249]:


model1.resid


# ## Sum of Squared Errors

# In[250]:


SSE = sum(model1.resid ** 2)
SSE


# In[251]:


RMSE = np.sqrt(SSE / len(wine))


# In[252]:


RMSE


# # Linear Regression (two variables)

# In[253]:


model2 = sm.ols('Price ~ AGST + HarvestRain', data=wine).fit()


# In[254]:


model2.summary()


# ## Sum of Squared Errors

# In[255]:


SSE = sum(model2.resid ** 2)


# In[256]:


SSE


# In[257]:


SST = sum((wine['Price'] - np.mean(wine['Price'])) ** 2)


# In[258]:


SST


# # Linear Regression (all variables)

# In[259]:


model3 = sm.ols('Price ~ AGST + HarvestRain + WinterRain + Age + FrancePop', data=wine).fit()


# In[260]:


model3.summary()


# ## Sum of Squared Errors

# In[261]:


SSE = sum(model3.resid ** 2)


# In[262]:


SSE


# # Quick Question

# In[263]:


model_quick = sm.ols('Price~ HarvestRain + WinterRain',data=wine).fit()


# In[264]:


model_quick.summary()


# In[265]:


model_quick.params


# # Remove FrancePop

# In[266]:


model4 = sm.ols('Price ~ AGST + HarvestRain+WinterRain+Age', data=wine).fit()


# In[267]:


model4.summary()


# In[268]:


model3.summary()


# In[269]:


import statsmodels.api as sm_api
table = sm_api.stats.anova_lm(model3,model4)


# In[270]:


table


# # Correlations

# In[271]:


np.corrcoef(wine['WinterRain'],wine['Price'])[0][1]


# In[273]:


from scipy.stats.stats import pearsonr
pearsonr(wine['WinterRain'],wine['Price'])[0]


# In[274]:


np.corrcoef(wine['Age'], wine['FrancePop'])[0][1]


# In[275]:


pd.DataFrame.corr(wine)


# In[276]:


model5 = sm.ols('Price ~ AGST + HarvestRain + WinterRain', data=wine).fit()


# In[277]:


model5.summary()


# In[278]:


model6 = sm.ols('Price ~ AGST + HarvestRain + WinterRain+ FrancePop', data=wine).fit()


# In[279]:


model6.summary()


# In[280]:


model7 = sm.ols('Price~Age+FrancePop',data=wine).fit()


# In[281]:


model7.summary()


# In[282]:


wineTest = pd.read_csv("https://storage.googleapis.com/dimensionless/Analytics/wine_test.csv")


# In[283]:


wineTest.dtypes


# In[284]:


predictTest = model4.predict(wineTest)


# In[285]:


predictTest


# In[286]:


wineTest['Price']


# # Compute R-squared

# In[287]:


SSE = sum((wineTest['Price'] - predictTest) ** 2)


# In[288]:


SST = sum((wineTest['Price'] - np.mean(wine['Price'])) ** 2)


# In[289]:


1 - SSE / SST


# # Prediction using  model2

# In[300]:


predict_model2 = model2.predict(wineTest)


# In[301]:


SSE = sum((wineTest['Price'] - predict_model2) ** 2)


# In[302]:


SST = sum((wineTest['Price'] - np.mean(wine['Price'])) ** 2)


# In[303]:


1 - SSE / SST


# In[304]:


wineTest.to_csv('wine_write.csv')


# # FeaturePlot

# In[305]:


fig, ax = plt.subplots(nrows=1,ncols=5,figsize=(15,2))

plt.subplot(151)
plt.plot(wine['WinterRain'] / 100,'ro')
plt.title('WinterRain')

plt.subplot(152)
plt.plot(wine['AGST'] / 100,'go')
plt.title('AGST')

plt.subplot(153)
plt.plot(wine['HarvestRain'] / 100,'bo')
plt.title('HarvestRain')

plt.subplot(154)
plt.plot(wine['Age'] / 100,'co')
plt.title('Age')

plt.subplot(155)
plt.plot(wine['FrancePop'] / 100,'ko')
plt.title('FrancePop')

fig.subplots_adjust(wspace=.6)


# # VIF

# In[159]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# In[171]:


def dmatrices_custom(target,features,dataframe):
    return dmatrices(target + '~' + features,dataframe,return_type='dataframe')


# In[176]:


def vif_factor(X_frame):
    return [variance_inflation_factor(X_frame.values, i) for i in range(X.shape[1])]


# In[226]:


def show_vif(X):
    vif = pd.DataFrame()
    vif["VIF Factor"] = vif_factor(X)
    vif["Features"] = X.columns
    return vif[vif.columns.tolist()[::-1]][1:].T


# In[172]:


target = 'Price' ## Common target variable


# ## Model-2  (Price ~ AGST + HarvestRain)

# In[228]:


data = wine[['Price','AGST','HarvestRain']]


# In[229]:


features = 'AGST+HarvestRain'


# In[230]:


y, X = dmatrices_custom(target,features,data)


# In[231]:


show_vif(X)


# ## Model - 3 (Price ~ AGST + HarvestRain + WinterRain + Age + FrancePop)

# In[232]:


data = wine[['Price','AGST','HarvestRain','WinterRain','Age','FrancePop']]


# In[233]:


features = 'AGST+HarvestRain+WinterRain+Age+FrancePop'


# In[234]:


y, X = dmatrices_custom(target,features,data)


# In[235]:


show_vif(X)


# ## Model - 4 (Price ~ AGST + HarvestRain + WinterRain + Age)

# In[236]:


data = wine[['Price','AGST','HarvestRain','WinterRain','Age']]


# In[237]:


features = 'AGST+HarvestRain+WinterRain+Age'


# In[238]:


y, X = dmatrices_custom(target,features,data)


# In[239]:


show_vif(X)


# ## Model - 5 (Price ~ AGST + HarvestRain + WinterRain)

# In[240]:


data = wine[['Price','AGST','HarvestRain','WinterRain']]


# In[241]:


features = 'AGST+HarvestRain+WinterRain'


# In[242]:


y, X = dmatrices_custom(target,features,data)


# In[243]:


show_vif(X)

