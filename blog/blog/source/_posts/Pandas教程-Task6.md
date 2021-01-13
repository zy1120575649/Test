---
title: Pandas教程-Task6
date: 2020-04-26 19:25:12
tags:
    - DataWhale
    - Pandas
categories: Pandas教程
---

```python
import pandas as pd
import numpy as np
df = pd.read_csv('Documents/Pandas教程/joyful-pandas-master/data/table_missing.csv')
df.head()
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
      <th>School</th>
      <th>Class</th>
      <th>ID</th>
      <th>Gender</th>
      <th>Address</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>NaN</td>
      <td>M</td>
      <td>street_1</td>
      <td>173</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>NaN</td>
      <td>F</td>
      <td>street_2</td>
      <td>192</td>
      <td>NaN</td>
      <td>32.5</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1103.0</td>
      <td>M</td>
      <td>street_2</td>
      <td>186</td>
      <td>NaN</td>
      <td>87.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S_1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>street_2</td>
      <td>167</td>
      <td>81.0</td>
      <td>80.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1105.0</td>
      <td>NaN</td>
      <td>street_4</td>
      <td>159</td>
      <td>64.0</td>
      <td>84.8</td>
      <td>A-</td>
    </tr>
  </tbody>
</table>
</div>



## 一、缺失观测及其类型


```python
#1. 了解缺失信息
#（a）isna和notna方法
#对Series使用会返回布尔列表
df['Physics'].isna().head()
```




    0    False
    1    False
    2    False
    3     True
    4    False
    Name: Physics, dtype: bool




```python
df['Physics'].notna().head()
```




    0     True
    1     True
    2     True
    3    False
    4     True
    Name: Physics, dtype: bool




```python
df.isna().head()
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
      <th>School</th>
      <th>Class</th>
      <th>ID</th>
      <th>Gender</th>
      <th>Address</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isna().sum()
```




    School      0
    Class       4
    ID          6
    Gender      7
    Address     0
    Height      0
    Weight     13
    Math        5
    Physics     4
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 35 entries, 0 to 34
    Data columns (total 9 columns):
    School     35 non-null object
    Class      31 non-null object
    ID         29 non-null float64
    Gender     28 non-null object
    Address    35 non-null object
    Height     35 non-null int64
    Weight     22 non-null float64
    Math       30 non-null float64
    Physics    31 non-null object
    dtypes: float64(3), int64(1), object(5)
    memory usage: 2.6+ KB



```python
df[df['Physics'].isna()]
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
      <th>School</th>
      <th>Class</th>
      <th>ID</th>
      <th>Gender</th>
      <th>Address</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>S_1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>street_2</td>
      <td>167</td>
      <td>81.0</td>
      <td>80.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>S_1</td>
      <td>C_2</td>
      <td>1204.0</td>
      <td>F</td>
      <td>street_5</td>
      <td>162</td>
      <td>63.0</td>
      <td>33.8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>S_1</td>
      <td>C_3</td>
      <td>1304.0</td>
      <td>NaN</td>
      <td>street_2</td>
      <td>195</td>
      <td>70.0</td>
      <td>85.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>S_2</td>
      <td>C_2</td>
      <td>2203.0</td>
      <td>M</td>
      <td>street_4</td>
      <td>155</td>
      <td>91.0</td>
      <td>73.8</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.notna().all(1)]
```

## 二、缺失数据的运算与分组


```python
#1. 加号与乘号规则
#使用加法时，缺失值为0
s = pd.Series([2,3,np.nan,4])
s.sum()
```




    9.0




```python
s.prod()
```




    24.0




```python
s.cumsum()
```




    0    2.0
    1    5.0
    2    NaN
    3    9.0
    dtype: float64




```python
s.cumprod()
```




    0     2.0
    1     6.0
    2     NaN
    3    24.0
    dtype: float64




```python
s.pct_change()
```




    0         NaN
    1    0.500000
    2    0.000000
    3    0.333333
    dtype: float64



## 三、填充与剔除


```python
#1. fillna方法
#（a）值填充与前后向填充（分别与ffill方法和bfill方法等价）
df['Physics'].fillna('missing').head()
```




    0         A+
    1         B+
    2         B+
    3    missing
    4         A-
    Name: Physics, dtype: object




```python
df['Physics'].fillna(method='ffill').head()
```




    0    A+
    1    B+
    2    B+
    3    B+
    4    A-
    Name: Physics, dtype: object




```python
df['Physics'].fillna(method='backfill').head()
```




    0    A+
    1    B+
    2    B+
    3    A-
    4    A-
    Name: Physics, dtype: object



## 四、插值（interpolation）


```python
#1. 线性插值
#（a）索引无关的线性插值
#默认状态下，interpolate会对缺失的值进行线性插值¶
```




    0     1.0
    1    10.0
    2    15.0
    3    -5.0
    4    -2.0
    5     NaN
    6     NaN
    7    28.0
    dtype: float64




```python
s.interpolate()
```




    0     1.0
    1    10.0
    2    15.0
    3    -5.0
    4    -2.0
    5     8.0
    6    18.0
    7    28.0
    dtype: float64




```python
s.index = np.sort(np.random.randint(50,300,8))
s.interpolate()
#值不变
```




    137     1.0
    148    10.0
    167    15.0
    171    -5.0
    171    -2.0
    283     8.0
    290    18.0
    299    28.0
    dtype: float64




```python

```
