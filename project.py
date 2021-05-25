#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


test_data = pd.read_csv('C:\경로/doge-usd-max.csv')
test_data


# In[3]:


pd.core.frame.DataFrame 


# In[4]:


test_data.plot.line() 


# In[5]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 


# In[6]:


test_data_11 = test_data.drop(['market_cap','snapped_at' ,'total_volume'], axis=1) 
test_data_11

test_data_22 = test_data_11[2500:2703] 
test_data_22.append(test_data_22,ignore_index = True) 

test_data_22.reset_index().rename(columns={"index":"id"}) 
test_data_33 = test_data_22.rename_axis('id').reset_index()
test_data_22.index.name='id'
test_data_44 = test_data_22.reset_index() 
test_data_44
test_data_22 = test_data_44


# In[7]:


test_data_22


# ## 여기부터 statsmodels 선형 회귀 분석

# In[8]:


from statsmodels.formula.api import ols


# In[9]:


fig = plt.figure(figsize=(8,8)) 
fig.set_facecolor('white') 
 
font_size = 15 
plt.scatter(test_data_22['id'],test_data_22['price']) 
 
plt.xlabel('id', fontsize=font_size) 
plt.ylabel('price',fontsize=font_size)
plt.show()


# In[10]:


fit = ols('price ~ id',data=test_data_22).fit()
fit.summary
print(fit.params.Intercept) 
print(fit.params.id)


# In[11]:


fit.fittedvalues


# In[12]:


fit.resid 


# In[13]:


fit.predict(exog=dict(id=[2800]))


# ## 여기부터 sklearn 선형 회귀 분석

# In[14]:


from sklearn.linear_model import LinearRegression


x = test_data_22['id'].values.reshape(-1,1)
y = test_data_22['price']
 
fit = LinearRegression().fit(x,y)


# In[15]:


print(fit.intercept_) 
print(fit.coef_)


# In[16]:


residual = y - fit.predict(x)
print(residual)


# In[17]:


fit.predict([[2800]])


# # 여기부터 tensorflow

# In[18]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) 
b = tf.Variable(tf.zeros([1])) 
y1 = W * x + b 

loss = tf.reduce_mean(tf.square(y1 - y)) 
optimizer = tf.train.GradientDescentOptimizer(0.000000005) 
train = optimizer.minimize(loss) 

init = tf.initialize_all_variables() 

sess = tf.Session() 
sess.run(init) 

for step in range(300):
     sess.run(train)
     if (step % 30 == 0):
         print(step, sess.run(W), sess.run(b))
         print(step, sess.run(loss))

predict_x = int(input('예측값 입력 : '))
print(sess.run(W)*predict_x + sess.run(b))

