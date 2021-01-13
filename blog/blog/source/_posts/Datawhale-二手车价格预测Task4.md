---
title: Datawhale-二手车价格预测Task4
date: 2020-03-30 00:07:17
tags:
    - DataWhale
    - 天池
categories: 数据挖掘
---

# Task4-建模调参
这部分的学习目标是：了解常用的机器学习模型，并掌握机器学习模型的建模与调参流程。
<!--more-->
## 1 内容介绍

1. 线性回归模型：
线性回归对于特征的要求；
处理长尾分布；
理解线性回归模型；

2. 模型性能验证：
评价函数与目标函数；
交叉验证方法；
留一验证方法；
针对时间序列问题的验证；
绘制学习率曲线；
绘制验证曲线；

3. 嵌入式特征选择：
Lasso回归；
Ridge回归；
决策树；

4. 模型对比：
常用线性模型；
常用非线性模型；

5. 模型调参：
贪心调参方法；
网格调参方法；
贝叶斯调参方法

## 2 代码示例
### 2.1读取数据 


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```


```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
```


```python
sample_feature = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))
```

    Memory usage of dataframe is 62099672.00 MB
    Memory usage after optimization is: 16520303.00 MB
    Decreased by 73.4%



```python
continuous_feature_names = [x for x in sample_feature.columns if x not in ['price','brand','model','brand']]
```

### 2.2线性回归 & 五折交叉验证 & 模拟真实业务情况


```python
sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)
train = sample_feature[continuous_feature_names + ['price']]

train_X = train[continuous_feature_names]
train_y = train['price']
```

#### 简单建模


```python
from sklearn.linear_model import LinearRegression
```


```python
model = LinearRegression(normalize=True)
```


```python
model = model.fit(train_X, train_y)
```

查看训练的线性回归模型的截距（intercept）与权重(coef)：


```python
'intercept:'+ str(model.intercept_)

sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)
```




    [('v_6', 3367064.3416419057),
     ('v_8', 700675.560939892),
     ('v_9', 170630.27723222843),
     ('v_7', 32322.661932036997),
     ('v_12', 20473.67079699449),
     ('v_3', 17868.079541513765),
     ('v_11', 11474.938996727224),
     ('v_13', 11261.76456002154),
     ('v_10', 2683.920090598985),
     ('gearbox', 881.822503924978),
     ('fuelType', 363.9042507217718),
     ('bodyType', 189.6027101206811),
     ('city', 44.94975120523234),
     ('power', 28.55390161675385),
     ('brand_price_median', 0.510372813407827),
     ('brand_price_std', 0.4503634709262942),
     ('brand_amount', 0.14881120395066458),
     ('brand_price_max', 0.0031910186703154246),
     ('SaleID', 5.355989919860341e-05),
     ('offerType', 1.085083931684494e-05),
     ('seller', -1.564621925354004e-07),
     ('train', -9.801937267184258e-06),
     ('brand_price_sum', -2.1750068681877473e-05),
     ('name', -0.00029800127130055304),
     ('used_time', -0.002515894332882901),
     ('brand_price_average', -0.40490484510106556),
     ('brand_price_min', -2.2467753486891016),
     ('power_bin', -34.420644117280766),
     ('v_14', -274.78411807803394),
     ('kilometer', -372.8975266607397),
     ('notRepairedDamage', -495.19038446291916),
     ('v_0', -2045.0549573549322),
     ('v_5', -11022.986240482482),
     ('v_4', -15121.731109861388),
     ('v_2', -26098.29992056384),
     ('v_1', -45556.18929727631)]




```python
from matplotlib import pyplot as plt
```


```python
subsample_index = np.random.randint(low=0, high=len(train_y), size=50)
```


```python
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price is obvious different from true price')
plt.show()
```

    The predicted price is obvious different from true price



![png](output_15_1.png)



```python
import seaborn as sns
print('It is clear to see the price shows a typical exponential distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y)
plt.subplot(1,2,2)
sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])
```

    It is clear to see the price shows a typical exponential distribution





    <matplotlib.axes._subplots.AxesSubplot at 0x1246a9f50>




![png](output_16_2.png)



```python
train_y_ln = np.log(train_y + 1)
```


```python
import seaborn as sns
print('The transformed price seems like normal distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y_ln)
plt.subplot(1,2,2)
sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
```

    The transformed price seems like normal distribution





    <matplotlib.axes._subplots.AxesSubplot at 0x1a2a8ca390>




![png](output_18_2.png)



```python
model = model.fit(train_X, train_y_ln)

print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)
```

    intercept:18.75074946555847





    [('v_9', 8.052409900567731),
     ('v_5', 5.764236596654289),
     ('v_12', 1.6182081236824377),
     ('v_1', 1.4798310582980099),
     ('v_11', 1.166901656362573),
     ('v_13', 0.9404711296038616),
     ('v_7', 0.713727308356608),
     ('v_3', 0.6837875771100316),
     ('v_0', 0.008500518010162533),
     ('power_bin', 0.008497969302895451),
     ('gearbox', 0.007922377278326771),
     ('fuelType', 0.006684769706822626),
     ('bodyType', 0.00452352009270304),
     ('power', 0.000716189420535552),
     ('brand_price_min', 3.334351114749525e-05),
     ('brand_amount', 2.897879704277287e-06),
     ('brand_price_median', 1.2571172873067898e-06),
     ('brand_price_std', 6.659176363472789e-07),
     ('brand_price_max', 6.194956307515058e-07),
     ('brand_price_average', 5.999345964992107e-07),
     ('SaleID', 2.1194170039649608e-08),
     ('offerType', 1.4733814168721437e-10),
     ('train', 1.0828671292983927e-11),
     ('seller', -6.533440455314121e-11),
     ('brand_price_sum', -1.5126504215917122e-10),
     ('name', -7.015512588913706e-08),
     ('used_time', -4.122479372350841e-06),
     ('city', -0.0022187824810423534),
     ('v_14', -0.004234223418139086),
     ('kilometer', -0.013835866226883525),
     ('notRepairedDamage', -0.2702794234984643),
     ('v_4', -0.8315701200997693),
     ('v_2', -0.9470842241656942),
     ('v_10', -1.626146668976213),
     ('v_8', -40.34300748761572),
     ('v_6', -238.79036385506865)]




```python
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price seems normal after np.log transforming')
plt.show()
```

    The predicted price seems normal after np.log transforming



![png](output_20_1.png)


#### 五折交叉验证

在使用训练集对参数进行训练的时候，经常会发现人们通常会将一整个训练集分为三个部分（比如mnist手写训练集）。一般分为：训练集（train_set），评估集（valid_set），测试集（test_set）这三个部分。这其实是为了保证训练效果而特意设置的。其中测试集很好理解，其实就是完全不参与训练的数据，仅仅用来观测测试效果的数据。而训练集和评估集则牵涉到下面的知识了。

因为在实际的训练中，训练的结果对于训练集的拟合程度通常还是挺好的（初始条件敏感），但是对于训练集之外的数据的拟合程度通常就不那么令人满意了。因此我们通常并不会把所有的数据集都拿来训练，而是分出一部分来（这一部分不参加训练）对训练集生成的参数进行测试，相对客观的判断这些参数对训练集之外的数据的符合程度。这种思想就称为交叉验证（Cross Validation）


```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer
```


```python
def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result
    return wrapper
```


```python
scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5, scoring=make_scorer(log_transfer(mean_absolute_error)))
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.8s finished



```python
print('AVG:', np.mean(scores))
```

    AVG: 1.3658023920314182



```python
scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.8s finished



```python
print('AVG:', np.mean(scores))
```

    AVG: 0.19325301837047415



```python
scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cv1</th>
      <th>cv2</th>
      <th>cv3</th>
      <th>cv4</th>
      <th>cv5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE</th>
      <td>0.190792</td>
      <td>0.193758</td>
      <td>0.194132</td>
      <td>0.191825</td>
      <td>0.195758</td>
    </tr>
  </tbody>
</table>
</div>



#### 模拟真实业务情况
但在事实上，由于我们并不具有预知未来的能力，五折交叉验证在某些与时间相关的数据集上反而反映了不真实的情况。通过2018年的二手车价格预测2017年的二手车价格，这显然是不合理的，因此我们还可以采用时间顺序对数据集进行分隔。在本例中，我们选用靠前时间的4/5样本当作训练集，靠后时间的1/5当作验证集，最终结果与五折交叉验证差距不大。


```python
import datetime
```


```python
sample_feature = sample_feature.reset_index(drop=True)
```


```python
split_point = len(sample_feature) // 5 * 4
```


```python
train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)
```


```python
model = model.fit(train_X, train_y_ln)
```


```python
mean_absolute_error(val_y_ln, model.predict(val_X))
```




    0.19577667270300972



#### 绘制学习率曲线与验证曲线


```python
from sklearn.model_selection import learning_curve, validation_curve
```


```python
? learning_curve
```


```python
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel('Training example')  
    plt.ylabel('score')  
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring = make_scorer(mean_absolute_error))  
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()#区域  
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                     train_scores_mean + train_scores_std, alpha=0.1,  
                     color="r")  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                     test_scores_mean + test_scores_std, alpha=0.1,  
                     color="g")  
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',  
             label="Training score")  
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",  
             label="Cross-validation score")  
    plt.legend(loc="best")  
    return plt  
```


```python
plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)  
```




    <module 'matplotlib.pyplot' from '/opt/anaconda3/lib/python3.7/site-packages/matplotlib/pyplot.py'>




![png](output_40_1.png)


#### 多种模型对比


```python
train = sample_feature[continuous_feature_names + ['price']].dropna()

train_X = train[continuous_feature_names]
train_y = train['price']
train_y_ln = np.log(train_y + 1)
```

##### 线性模型 & 嵌入式特征选择


```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
```


```python
models = [LinearRegression(),
          Ridge(),
          Lasso()]
```


```python
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')
```

    LinearRegression is finished
    Ridge is finished
    Lasso is finished



```python
result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LinearRegression</th>
      <th>Ridge</th>
      <th>Lasso</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cv1</th>
      <td>0.190792</td>
      <td>0.194832</td>
      <td>0.383899</td>
    </tr>
    <tr>
      <th>cv2</th>
      <td>0.193758</td>
      <td>0.197632</td>
      <td>0.381893</td>
    </tr>
    <tr>
      <th>cv3</th>
      <td>0.194132</td>
      <td>0.198123</td>
      <td>0.384090</td>
    </tr>
    <tr>
      <th>cv4</th>
      <td>0.191825</td>
      <td>0.195670</td>
      <td>0.380526</td>
    </tr>
    <tr>
      <th>cv5</th>
      <td>0.195758</td>
      <td>0.199676</td>
      <td>0.383611</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = LinearRegression().fit(train_X, train_y_ln)
print('intercept:'+ str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
```

    intercept:18.747877090866396





    <matplotlib.axes._subplots.AxesSubplot at 0x122af1510>




![png](output_48_2.png)


L2正则化在拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。可以设想一下对于一个线性回归方程，若参数很大，那么只要数据偏移一点点，就会对结果造成很大的影响；但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响，专业一点的说法是『抗扰动能力强』


```python
model = Ridge().fit(train_X, train_y_ln)
print('intercept:'+ str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
```

    intercept:4.6717097880636445





    <matplotlib.axes._subplots.AxesSubplot at 0x122afafd0>




![png](output_50_2.png)


L1正则化有助于生成一个稀疏权值矩阵，进而可以用于特征选择。如下图，我们发现power与userd_time特征非常重要。


```python
model = Lasso().fit(train_X, train_y_ln)
print('intercept:'+ str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
```

    intercept:8.672182402894254





    <matplotlib.axes._subplots.AxesSubplot at 0x12483b050>




![png](output_52_2.png)


##### 非线性模型

####  模型调参


```python
## LGB的参数集合：

objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3,5,10,15,20,40, 55]
max_depth = [3,5,10,15,20,40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []
```

#####  贪心调参


```python
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score
    
best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0], num_leaves=leaves)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score
    
best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x:x[1])[0],
                          max_depth=depth)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score
```


```python
sns.lineplot(x=['0_initial','1_turning_obj','2_turning_leaves','3_turning_depth'], y=[0.143 ,min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])
```

#####  Grid Search 调参


```python
from sklearn.model_selection import GridSearchCV
```


```python
parameters = {'objective': objective , 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)
clf.best_params_
model = LGBMRegressor(objective='regression',
                          num_leaves=55,
                          max_depth=15)
np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
```

#####  贝叶斯调参


```python
from bayes_opt import BayesianOptimization
```


```python
def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective = 'regression_l1',
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample = subsample,
            min_child_samples = int(min_child_samples)
        ),
        X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val
```


```python
rf_bo = BayesianOptimization(
    rf_cv,
    {
    'num_leaves': (2, 100),
    'max_depth': (2, 100),
    'subsample': (0.1, 1),
    'min_child_samples' : (2, 100)
    }
)
rf_bo.maximize()
1 - rf_bo.max['target']
```
