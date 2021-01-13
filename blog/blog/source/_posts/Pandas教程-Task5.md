---
title: Pandas教程-Task5
date: 2020-04-26 19:25:07
tags:
    - DataWhale
    - Pandas
categories: Pandas教程
---

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



## 一、append与assign


```python
#1. append方法
#（a）利用序列添加行（必须指定name）
df_append = df.loc[:3,['Gender','Height']].copy()
df_append
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
      <th>Gender</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>173</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>186</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>167</td>
    </tr>
  </tbody>
</table>
</div>




```python
s = pd.Series({'Gender':'F','Height':188},name='new_row')
df_append.append(s)
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
      <th>Gender</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>173</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>186</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>167</td>
    </tr>
    <tr>
      <th>new_row</th>
      <td>F</td>
      <td>188</td>
    </tr>
  </tbody>
</table>
</div>




```python
#（b）用DataFrame添加表
df_temp = pd.DataFrame({'Gender':['F','M'],'Height':[188,176]},index=['new_1','new_2'])
df_append.append(df_temp)
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
      <th>Gender</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>173</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>186</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>167</td>
    </tr>
    <tr>
      <th>new_1</th>
      <td>F</td>
      <td>188</td>
    </tr>
    <tr>
      <th>new_2</th>
      <td>M</td>
      <td>176</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2. assign方法
#该方法主要用于添加列，列名直接由参数指定
s = pd.Series(list('abcd'),index=range(4))
df_append.assign(Letter=s)
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
      <th>Gender</th>
      <th>Height</th>
      <th>Letter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>173</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>192</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>186</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>167</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>



## 二、combine与update


```python
#1. comine方法
#comine和update都是用于表的填充函数，可以根据某种规则填充
df_combine_1 = df.loc[:1,['Gender','Height']].copy()
df_combine_2 = df.loc[10:11,['Gender','Height']].copy()
df_combine_1.combine(df_combine_2,lambda x,y:print(x,y))
```

    0       M
    1       F
    10    NaN
    11    NaN
    Name: Gender, dtype: object 0     NaN
    1     NaN
    10      M
    11      F
    Name: Gender, dtype: object
    0     173.0
    1     192.0
    10      NaN
    11      NaN
    Name: Height, dtype: float64 0       NaN
    1       NaN
    10    161.0
    11    175.0
    Name: Height, dtype: float64





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
      <th>Gender</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2. update方法
#（a）三个特点
#①返回的框索引只会与被调用框的一致（默认使用左连接，下一节会介绍）
#②第二个框中的nan元素不会起作用
#③没有返回值，直接在df上操作
df1 = pd.DataFrame({'A': [1, 2, 3],
                    'B': [400, 500, 600]})
df2 = pd.DataFrame({'B': [4, 5, 6],
                    'C': [7, 8, 9]})
df1.update(df2)
df1
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## 三、concat方法


```python
#concat方法可以在两个维度上拼接，默认纵向凭借（axis=0），拼接方式默认外连接
#所谓外连接，就是取拼接方向的并集，而'inner'时取拼接方向（若使用默认的纵向拼接，则为列的交集）的交集
#下面举一些例子说明其参数：
df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                    index = [0,1])
df2 = pd.DataFrame({'A': ['A2', 'A3'],
                    'B': ['B2', 'B3']},
                    index = [2,3])
df3 = pd.DataFrame({'A': ['A1', 'A3'],
                    'D': ['D1', 'D3'],
                    'E': ['E1', 'E3']},
                    index = [1,3])
```


```python
pd.concat([df1,df2])
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([df1,df2],axis=1)
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
      <th>A</th>
      <th>B</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([df3,df1],join='inner')
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
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>A0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
    </tr>
  </tbody>
</table>
</div>



## 四、merge与join


```python
#1. merge函数
#merge函数的作用是将两个pandas对象横向合并，遇到重复的索引项时会使用笛卡尔积，默认inner连接，可选left、outer、right连接
#所谓左连接，就是指以第一个表索引为基准，右边的表中如果不再左边的则不加入，如果在左边的就以笛卡尔积的方式加入
#merge/join与concat的不同之处在于on参数，可以指定某一个对象为key来进行连接
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']}) 
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
right2 = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3']})
```


```python
pd.merge(left, right, on='key1')
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
      <th>key1</th>
      <th>key2_x</th>
      <th>A</th>
      <th>B</th>
      <th>key2_y</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>K0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K0</td>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
      <td>K0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>K0</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>K0</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>K2</td>
      <td>K1</td>
      <td>A3</td>
      <td>B3</td>
      <td>K0</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2. join函数
#join函数作用是将多个pandas对象横向拼接，
#遇到重复的索引项时会使用笛卡尔积，默认左连接，可选inner、outer、right连接
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                    index=['K0', 'K2', 'K3'])
left.join(right)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
  </tbody>
</table>
</div>


