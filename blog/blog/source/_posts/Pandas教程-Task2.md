---
title: Pandas教程-Task2
date: 2020-04-23 15:03:31
tags:
    - DataWhale
    - Pandas
categories: Pandas教程
---

```python
import numpy as np
import pandas as pd
df = pd.read_csv('Documents/Pandas教程/joyful-pandas-master/data/table.csv',index_col='ID')
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
      <th>Gender</th>
      <th>Address</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1101</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>M</td>
      <td>street_1</td>
      <td>173</td>
      <td>63</td>
      <td>34.0</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>F</td>
      <td>street_2</td>
      <td>192</td>
      <td>73</td>
      <td>32.5</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>M</td>
      <td>street_2</td>
      <td>186</td>
      <td>82</td>
      <td>87.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>F</td>
      <td>street_2</td>
      <td>167</td>
      <td>81</td>
      <td>80.4</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>S_1</td>
      <td>C_1</td>
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



## 单级索引
###  loc方法、iloc方法、[]操作符
最常用的索引方法可能就是这三类，其中iloc表示位置索引，loc表示标签索引，[]也具有很大的便利性，各有特点
### 布尔索引
### 快速标量索引
### 区间索引

## 多级索引
1.创建多级索引
（a）通过from_tuple或from_arrays


```python
tuples = [('A','a'),('A','b'),('B','a'),('B','b')]
mul_index = pd.MultiIndex.from_tuples(tuples, names=('Upper', 'Lower'))
mul_index
```




    MultiIndex([('A', 'a'),
                ('A', 'b'),
                ('B', 'a'),
                ('B', 'b')],
               names=['Upper', 'Lower'])




```python
pd.DataFrame({'Score':['perfect','good','fair','bad']},index=mul_index)
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
      <th></th>
      <th>Score</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>a</th>
      <td>perfect</td>
    </tr>
    <tr>
      <th>b</th>
      <td>good</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>a</th>
      <td>fair</td>
    </tr>
    <tr>
      <th>b</th>
      <td>bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
L1 = list('AABB')
L2 = list('abab')
tuples = list(zip(L1,L2))
mul_index = pd.MultiIndex.from_tuples(tuples, names=('Upper', 'Lower'))
pd.DataFrame({'Score':['perfect','good','fair','bad']},index=mul_index)
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
      <th></th>
      <th>Score</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>a</th>
      <td>perfect</td>
    </tr>
    <tr>
      <th>b</th>
      <td>good</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>a</th>
      <td>fair</td>
    </tr>
    <tr>
      <th>b</th>
      <td>bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
arrays = [['A','a'],['A','b'],['B','a'],['B','b']]
mul_index = pd.MultiIndex.from_tuples(arrays, names=('Upper', 'Lower'))
pd.DataFrame({'Score':['perfect','good','fair','bad']},index=mul_index)
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
      <th></th>
      <th>Score</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>a</th>
      <td>perfect</td>
    </tr>
    <tr>
      <th>b</th>
      <td>good</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>a</th>
      <td>fair</td>
    </tr>
    <tr>
      <th>b</th>
      <td>bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
mul_index
#由此看出内部自动转成元组
```




    MultiIndex([('A', 'a'),
                ('A', 'b'),
                ('B', 'a'),
                ('B', 'b')],
               names=['Upper', 'Lower'])



（b）通过from_product


```python
L1 = ['A','B']
L2 = ['a','b']
pd.MultiIndex.from_product([L1,L2],names=('Upper', 'Lower'))
#两两相乘
```

（c）指定df中的列创建（set_index方法）


```python
df_using_mul = df.set_index(['Class','Address'])
df_using_mul.head()
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
      <th></th>
      <th>School</th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
    <tr>
      <th>Class</th>
      <th>Address</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">C_1</th>
      <th>street_1</th>
      <td>S_1</td>
      <td>M</td>
      <td>173</td>
      <td>63</td>
      <td>34.0</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>street_2</th>
      <td>S_1</td>
      <td>F</td>
      <td>192</td>
      <td>73</td>
      <td>32.5</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>street_2</th>
      <td>S_1</td>
      <td>M</td>
      <td>186</td>
      <td>82</td>
      <td>87.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>street_2</th>
      <td>S_1</td>
      <td>F</td>
      <td>167</td>
      <td>81</td>
      <td>80.4</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>street_4</th>
      <td>S_1</td>
      <td>F</td>
      <td>159</td>
      <td>64</td>
      <td>84.8</td>
      <td>B+</td>
    </tr>
  </tbody>
</table>
</div>



2.多层索引切片


```python
df_using_mul.head()
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
      <th></th>
      <th>School</th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
    <tr>
      <th>Class</th>
      <th>Address</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">C_1</th>
      <th>street_1</th>
      <td>S_1</td>
      <td>M</td>
      <td>173</td>
      <td>63</td>
      <td>34.0</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>street_2</th>
      <td>S_1</td>
      <td>F</td>
      <td>192</td>
      <td>73</td>
      <td>32.5</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>street_2</th>
      <td>S_1</td>
      <td>M</td>
      <td>186</td>
      <td>82</td>
      <td>87.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>street_2</th>
      <td>S_1</td>
      <td>F</td>
      <td>167</td>
      <td>81</td>
      <td>80.4</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>street_4</th>
      <td>S_1</td>
      <td>F</td>
      <td>159</td>
      <td>64</td>
      <td>84.8</td>
      <td>B+</td>
    </tr>
  </tbody>
</table>
</div>



3.多层索引中的slice对象


```python
L1,L2 = ['A','B'],['a','b','c']
mul_index1 = pd.MultiIndex.from_product([L1,L2],names=('Upper', 'Lower'))
L3,L4 = ['D','E','F'],['d','e','f']
mul_index2 = pd.MultiIndex.from_product([L3,L4],names=('Big', 'Small'))
df_s = pd.DataFrame(np.random.rand(6,9),index=mul_index1,columns=mul_index2)
df_s
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Big</th>
      <th colspan="3" halign="left">D</th>
      <th colspan="3" halign="left">E</th>
      <th colspan="3" halign="left">F</th>
    </tr>
    <tr>
      <th></th>
      <th>Small</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">A</th>
      <th>a</th>
      <td>0.693152</td>
      <td>0.672084</td>
      <td>0.929580</td>
      <td>0.348846</td>
      <td>0.998656</td>
      <td>0.618890</td>
      <td>0.134470</td>
      <td>0.629302</td>
      <td>0.755844</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.793648</td>
      <td>0.663314</td>
      <td>0.793556</td>
      <td>0.632018</td>
      <td>0.837149</td>
      <td>0.573710</td>
      <td>0.083993</td>
      <td>0.774272</td>
      <td>0.634520</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.485607</td>
      <td>0.429198</td>
      <td>0.971712</td>
      <td>0.471989</td>
      <td>0.250029</td>
      <td>0.027371</td>
      <td>0.377319</td>
      <td>0.721766</td>
      <td>0.911952</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">B</th>
      <th>a</th>
      <td>0.434036</td>
      <td>0.430081</td>
      <td>0.590865</td>
      <td>0.106680</td>
      <td>0.678473</td>
      <td>0.946369</td>
      <td>0.681991</td>
      <td>0.788171</td>
      <td>0.272536</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.116107</td>
      <td>0.186889</td>
      <td>0.252728</td>
      <td>0.137849</td>
      <td>0.078398</td>
      <td>0.986727</td>
      <td>0.533018</td>
      <td>0.049788</td>
      <td>0.078079</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.577150</td>
      <td>0.858875</td>
      <td>0.468748</td>
      <td>0.016894</td>
      <td>0.717600</td>
      <td>0.283764</td>
      <td>0.595257</td>
      <td>0.037818</td>
      <td>0.839023</td>
    </tr>
  </tbody>
</table>
</div>


