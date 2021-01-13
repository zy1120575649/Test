---
title: Pandas教程-Task4
date: 2020-04-26 19:25:01
tags:
    - DataWhale
    - Pandas
categories: Pandas教程
---

第4章 变形


```python
import numpy as np
import pandas as pd
df = pd.read_csv('Documents/Pandas教程/joyful-pandas-master/data/table.csv')
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
      <td>1101</td>
      <td>M</td>
      <td>street_1</td>
      <td>173</td>
      <td>63</td>
      <td>34.0</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1102</td>
      <td>F</td>
      <td>street_2</td>
      <td>192</td>
      <td>73</td>
      <td>32.5</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1103</td>
      <td>M</td>
      <td>street_2</td>
      <td>186</td>
      <td>82</td>
      <td>87.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1104</td>
      <td>F</td>
      <td>street_2</td>
      <td>167</td>
      <td>81</td>
      <td>80.4</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1105</td>
      <td>F</td>
      <td>street_4</td>
      <td>159</td>
      <td>64</td>
      <td>84.8</td>
      <td>B+</td>
    </tr>
  </tbody>
</table>
</div>




```python
#1. pivot¶
#一般状态下，数据在DataFrame会以压缩（stacked）状态存放，例如上面的Gender，
#两个类别被叠在一列中，pivot函数可将某一列作为新的cols
df.pivot(index='ID',columns='Gender',values='Height').head()
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
      <th>Gender</th>
      <th>F</th>
      <th>M</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1101</th>
      <td>NaN</td>
      <td>173.0</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>192.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>NaN</td>
      <td>186.0</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>167.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>159.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2. pivot_table
#由于功能更多，速度上自然是比不上原来的pivot函数
%timeit df.pivot(index='ID',columns='Gender',values='Height')
%timeit pd.pivot_table(df,index='ID',columns='Gender',values='Height')
```

    3.15 ms ± 336 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    14 ms ± 663 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
#3. crosstab（交叉表）
#交叉表是一种特殊的透视表，典型的用途如分组统计，如现在想要统计关于街道和性别分组的频数：¶
pd.crosstab(index=df['Address'],columns=df['Gender'])
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
      <th>Gender</th>
      <th>F</th>
      <th>M</th>
    </tr>
    <tr>
      <th>Address</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>street_1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>street_2</th>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>street_4</th>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>street_5</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>street_6</th>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>street_7</th>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
#其他变形方法
#1. melt
#melt函数可以认为是pivot函数的逆操作，将unstacked状态的数据，压缩成stacked，使“宽”的DataFrame变“窄”
df_m = df[['ID','Gender','Math']]
df_m.head()
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
      <th>ID</th>
      <th>Gender</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1101</td>
      <td>M</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1102</td>
      <td>F</td>
      <td>32.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1103</td>
      <td>M</td>
      <td>87.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1104</td>
      <td>F</td>
      <td>80.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1105</td>
      <td>F</td>
      <td>84.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pivot(index='ID',columns='Gender',values='Math').head()
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
      <th>Gender</th>
      <th>F</th>
      <th>M</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1101</th>
      <td>NaN</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>32.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>NaN</td>
      <td>87.2</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>80.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>84.8</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivoted = df.pivot(index='ID',columns='Gender',values='Math')
result = pivoted.reset_index().melt(id_vars=['ID'],value_vars=['F','M'],value_name='Math')\
                     .dropna().set_index('ID').sort_index()
#检验是否与展开前的df相同，可以分别将这些链式方法的中间步骤展开，看看是什么结果
result.equals(df_m.set_index('ID'))
```




    True




```python
#哑变量与因子化
#1. Dummy Variable（哑变量）
#这里主要介绍get_dummies函数，其功能主要是进行one-hot编码
df_d = df[['Class','Gender','Weight']]
df_d.head()
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
      <th>Class</th>
      <th>Gender</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_1</td>
      <td>M</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_1</td>
      <td>F</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_1</td>
      <td>M</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_1</td>
      <td>F</td>
      <td>81</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_1</td>
      <td>F</td>
      <td>64</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.get_dummies(df_d[['Class','Gender']]).join(df_d['Weight']).head()
#可选prefix参数添加前缀，prefix_sep添加分隔符
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
      <th>Class_C_1</th>
      <th>Class_C_2</th>
      <th>Class_C_3</th>
      <th>Class_C_4</th>
      <th>Gender_F</th>
      <th>Gender_M</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>81</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>64</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2. factorize方法
#该方法主要用于自然数编码，并且缺失值会被记做-1，其中sort参数表示是否排序后赋值
codes, uniques = pd.factorize(['b', None, 'a', 'c', 'b'], sort=True)
display(codes)
display(uniques)
```


    array([ 1, -1,  0,  2,  1])



    array(['a', 'b', 'c'], dtype=object)

