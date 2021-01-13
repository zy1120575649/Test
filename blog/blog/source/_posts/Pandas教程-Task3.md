---
title: Pandas教程-Task3
date: 2020-04-25 16:53:12
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



## 一、SAC过程
### 1. 内涵
SAC指的是分组操作中的split-apply-combine过程
其中split指基于某一些规则，将数据拆成若干组，apply是指对每一组独立地使用函数，combine指将每一组的结果组合成某一类数据结构
### 2. apply过程
在该过程中，我们实际往往会遇到四类问题：
整合（Aggregation）——即分组计算统计量（如求均值、求每组元素个数）
变换（Transformation）——即分组对每个单元的数据进行操作（如元素标准化）
过滤（Filtration）——即按照某些规则筛选出一些组（如选出组内某一指标小于50的组）
综合问题——即前面提及的三种问题的混合

## 二、groupby函数


```python
grouped_single = df.groupby('School')
```


```python
grouped_single.get_group('S_1').head()
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




```python
grouped_mul = df.groupby(['School','Class'])
grouped_mul.get_group(('S_2','C_4'))
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
      <th>2401</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>F</td>
      <td>street_2</td>
      <td>192</td>
      <td>62</td>
      <td>45.3</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2402</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>M</td>
      <td>street_7</td>
      <td>166</td>
      <td>82</td>
      <td>48.7</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2403</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>F</td>
      <td>street_6</td>
      <td>158</td>
      <td>60</td>
      <td>59.7</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>2404</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>F</td>
      <td>street_2</td>
      <td>160</td>
      <td>84</td>
      <td>67.7</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>F</td>
      <td>street_6</td>
      <td>193</td>
      <td>54</td>
      <td>47.6</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_single.size()
```




    School
    S_1    15
    S_2    20
    dtype: int64




```python
grouped_mul.size()
```




    School  Class
    S_1     C_1      5
            C_2      5
            C_3      5
    S_2     C_1      5
            C_2      5
            C_3      5
            C_4      5
    dtype: int64




```python
grouped_single.ngroups
```




    2




```python
grouped_mul.ngroups

```




    7




```python
for name,group in grouped_single:
    print(name)
    display(group.head())
```

    S_1



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


    S_2



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
      <th>2101</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>M</td>
      <td>street_7</td>
      <td>174</td>
      <td>84</td>
      <td>83.3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2102</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>F</td>
      <td>street_6</td>
      <td>161</td>
      <td>61</td>
      <td>50.6</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>2103</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>M</td>
      <td>street_4</td>
      <td>157</td>
      <td>61</td>
      <td>52.5</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>2104</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>F</td>
      <td>street_5</td>
      <td>159</td>
      <td>97</td>
      <td>72.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>2105</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>M</td>
      <td>street_4</td>
      <td>170</td>
      <td>81</td>
      <td>34.2</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.set_index(['Gender','School']).groupby(level=1,axis=0).get_group('S_1').head()
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
      <th>Class</th>
      <th>Address</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>School</th>
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
      <th>M</th>
      <th>S_1</th>
      <td>C_1</td>
      <td>street_1</td>
      <td>173</td>
      <td>63</td>
      <td>34.0</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>F</th>
      <th>S_1</th>
      <td>C_1</td>
      <td>street_2</td>
      <td>192</td>
      <td>73</td>
      <td>32.5</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>M</th>
      <th>S_1</th>
      <td>C_1</td>
      <td>street_2</td>
      <td>186</td>
      <td>82</td>
      <td>87.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">F</th>
      <th>S_1</th>
      <td>C_1</td>
      <td>street_2</td>
      <td>167</td>
      <td>81</td>
      <td>80.4</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>S_1</th>
      <td>C_1</td>
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
print([attr for attr in dir(grouped_single) if not attr.startswith('_')])
```

    ['Address', 'Class', 'Gender', 'Height', 'Math', 'Physics', 'School', 'Weight', 'agg', 'aggregate', 'all', 'any', 'apply', 'backfill', 'bfill', 'boxplot', 'corr', 'corrwith', 'count', 'cov', 'cumcount', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'dtypes', 'expanding', 'ffill', 'fillna', 'filter', 'first', 'get_group', 'groups', 'head', 'hist', 'idxmax', 'idxmin', 'indices', 'last', 'mad', 'max', 'mean', 'median', 'min', 'ndim', 'ngroup', 'ngroups', 'nth', 'nunique', 'ohlc', 'pad', 'pct_change', 'pipe', 'plot', 'prod', 'quantile', 'rank', 'resample', 'rolling', 'sem', 'shift', 'size', 'skew', 'std', 'sum', 'tail', 'take', 'transform', 'tshift', 'var']



```python
grouped_single.head(2)
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
      <th>2101</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>M</td>
      <td>street_7</td>
      <td>174</td>
      <td>84</td>
      <td>83.3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2102</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>F</td>
      <td>street_6</td>
      <td>161</td>
      <td>61</td>
      <td>50.6</td>
      <td>B+</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_single.first()
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
      <th>Address</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
    <tr>
      <th>School</th>
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
      <th>S_1</th>
      <td>C_1</td>
      <td>M</td>
      <td>street_1</td>
      <td>173</td>
      <td>63</td>
      <td>34.0</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>S_2</th>
      <td>C_1</td>
      <td>M</td>
      <td>street_7</td>
      <td>174</td>
      <td>84</td>
      <td>83.3</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(np.random.choice(['a','b','c'],df.shape[0])).get_group('a').head()
#相当于将np.random.choice(['a','b','c'],df.shape[0])当做新的一列进行分组
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
      <th>1202</th>
      <td>S_1</td>
      <td>C_2</td>
      <td>F</td>
      <td>street_4</td>
      <td>176</td>
      <td>94</td>
      <td>63.5</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>S_1</td>
      <td>C_3</td>
      <td>M</td>
      <td>street_4</td>
      <td>161</td>
      <td>68</td>
      <td>31.5</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>2101</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>M</td>
      <td>street_7</td>
      <td>174</td>
      <td>84</td>
      <td>83.3</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[:5].groupby(lambda x:print(x)).head(0)
```

    1101
    1102
    1103
    1104
    1105





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
  </tbody>
</table>
</div>




```python
df.groupby(lambda x:'奇数行' if not df.index.get_loc(x)%2==1 else '偶数行').groups
```




    {'偶数行': Int64Index([1102, 1104, 1201, 1203, 1205, 1302, 1304, 2101, 2103, 2105, 2202,
                 2204, 2301, 2303, 2305, 2402, 2404],
                dtype='int64', name='ID'),
     '奇数行': Int64Index([1101, 1103, 1105, 1202, 1204, 1301, 1303, 1305, 2102, 2104, 2201,
                 2203, 2205, 2302, 2304, 2401, 2403, 2405],
                dtype='int64', name='ID')}




```python
math_score = df.set_index(['Gender','School'])['Math'].sort_index()
grouped_score = df.set_index(['Gender','School']).sort_index().\
            groupby(lambda x:(x,'均分及格' if math_score[x].mean()>=60 else '均分不及格'))
for name,_ in grouped_score:print(name)
```

    (('F', 'S_1'), '均分及格')
    (('F', 'S_2'), '均分及格')
    (('M', 'S_1'), '均分及格')
    (('M', 'S_2'), '均分不及格')


## 三、聚合、过滤和变换
### 1. 聚合（Aggregation）
所谓聚合就是把一堆数，变成一个标量，因此mean/sum/size/count/std/var/sem/describe/first/last/nth/min/max都是聚合函数
### 2. 过滤（Filteration）
filter函数是用来筛选某些组的（务必记住结果是组的全体），因此传入的值应当是布尔标量
### 3. 变换（Transformation）

