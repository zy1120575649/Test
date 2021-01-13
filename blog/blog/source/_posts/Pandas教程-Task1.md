---
title: Pandas教程-Task1
date: 2020-04-20 21:12:34
tags:
    - DataWhale
    - Pandas
categories: Pandas教程
---

```python
import pandas as pd
import numpy as np
```


```python
pd.__version__
```




    '0.25.3'




```python
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
df_txt = pd.read_table('Documents/Pandas教程/joyful-pandas-master/data/table.txt')
df_txt
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_excel = pd.read_excel('Documents/Pandas教程/joyful-pandas-master/data/table.xlsx')
df_excel.head()
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
df.to_csv('Documents/Pandas教程/joyful-pandas-master/data/new_table.csv')
```


```python
df.to_excel('Documents/Pandas教程/joyful-pandas-master/data/new_table2.xlsx', sheet_name='Sheet1')
```


```python
s = pd.Series(np.random.randn(5),index=['a','b','c','d','e'],name='这是一个Series',dtype='float64')
s
```




    a   -0.699435
    b   -0.583320
    c   -0.526598
    d   -1.110295
    e   -0.006360
    Name: 这是一个Series, dtype: float64




```python
s.values
```




    array([-0.6994351 , -0.5833203 , -0.52659774, -1.11029458, -0.00635964])




```python
s.name
```




    '这是一个Series'




```python
s.index
```




    Index(['a', 'b', 'c', 'd', 'e'], dtype='object')




```python
s.dtype
```




    dtype('float64')




```python
s['a']
```




    -0.6994350999787233




```python
s.mean()
```




    -0.585201471154681




```python
print([attr for attr in dir(s) if not attr.startswith('_')])
```

    ['T', 'a', 'abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 'align', 'all', 'any', 'append', 'apply', 'argmax', 'argmin', 'argsort', 'array', 'as_matrix', 'asfreq', 'asof', 'astype', 'at', 'at_time', 'autocorr', 'axes', 'b', 'between', 'between_time', 'bfill', 'bool', 'c', 'clip', 'clip_lower', 'clip_upper', 'combine', 'combine_first', 'compound', 'compress', 'copy', 'corr', 'count', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'd', 'describe', 'diff', 'div', 'divide', 'divmod', 'dot', 'drop', 'drop_duplicates', 'droplevel', 'dropna', 'dtype', 'dtypes', 'duplicated', 'e', 'empty', 'eq', 'equals', 'ewm', 'expanding', 'explode', 'factorize', 'ffill', 'fillna', 'filter', 'first', 'first_valid_index', 'floordiv', 'from_array', 'ge', 'get', 'get_dtype_counts', 'get_ftype_counts', 'get_values', 'groupby', 'gt', 'hasnans', 'head', 'hist', 'iat', 'idxmax', 'idxmin', 'iloc', 'index', 'infer_objects', 'interpolate', 'is_monotonic', 'is_monotonic_decreasing', 'is_monotonic_increasing', 'is_unique', 'isin', 'isna', 'isnull', 'item', 'items', 'iteritems', 'keys', 'kurt', 'kurtosis', 'last', 'last_valid_index', 'le', 'loc', 'lt', 'mad', 'map', 'mask', 'max', 'mean', 'median', 'memory_usage', 'min', 'mod', 'mode', 'mul', 'multiply', 'name', 'nbytes', 'ndim', 'ne', 'nlargest', 'nonzero', 'notna', 'notnull', 'nsmallest', 'nunique', 'pct_change', 'pipe', 'plot', 'pop', 'pow', 'prod', 'product', 'ptp', 'put', 'quantile', 'radd', 'rank', 'ravel', 'rdiv', 'rdivmod', 'reindex', 'reindex_like', 'rename', 'rename_axis', 'reorder_levels', 'repeat', 'replace', 'resample', 'reset_index', 'rfloordiv', 'rmod', 'rmul', 'rolling', 'round', 'rpow', 'rsub', 'rtruediv', 'sample', 'searchsorted', 'sem', 'set_axis', 'shape', 'shift', 'size', 'skew', 'slice_shift', 'sort_index', 'sort_values', 'squeeze', 'std', 'sub', 'subtract', 'sum', 'swapaxes', 'swaplevel', 'tail', 'take', 'to_clipboard', 'to_csv', 'to_dense', 'to_dict', 'to_excel', 'to_frame', 'to_hdf', 'to_json', 'to_latex', 'to_list', 'to_msgpack', 'to_numpy', 'to_period', 'to_pickle', 'to_sparse', 'to_sql', 'to_string', 'to_timestamp', 'to_xarray', 'transform', 'transpose', 'truediv', 'truncate', 'tshift', 'tz_convert', 'tz_localize', 'unique', 'unstack', 'update', 'value_counts', 'values', 'var', 'view', 'where', 'xs']



```python
df = pd.DataFrame({'col1':list('abcde'),'col2':range(5,10),'col3':[1.3,2.5,3.6,4.6,5.8]},
                 index=list('一二三四五'))
df
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>一</th>
      <td>a</td>
      <td>5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>二</th>
      <td>b</td>
      <td>6</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>三</th>
      <td>c</td>
      <td>7</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>四</th>
      <td>d</td>
      <td>8</td>
      <td>4.6</td>
    </tr>
    <tr>
      <th>五</th>
      <td>e</td>
      <td>9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['col1']
```




    一    a
    二    b
    三    c
    四    d
    五    e
    Name: col1, dtype: object




```python
type(df)
```




    pandas.core.frame.DataFrame




```python
type(df['col1'])
```




    pandas.core.series.Series




```python
df.rename(index={'一':'one'},columns={'col1':'new_col1'})
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
      <th>new_col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>a</td>
      <td>5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>二</th>
      <td>b</td>
      <td>6</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>三</th>
      <td>c</td>
      <td>7</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>四</th>
      <td>d</td>
      <td>8</td>
      <td>4.6</td>
    </tr>
    <tr>
      <th>五</th>
      <td>e</td>
      <td>9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    Index(['一', '二', '三', '四', '五'], dtype='object')




```python
df.columns
```




    Index(['col1', 'col2', 'col3'], dtype='object')




```python
df.values
```




    array([['a', 5, 1.3],
           ['b', 6, 2.5],
           ['c', 7, 3.6],
           ['d', 8, 4.6],
           ['e', 9, 5.8]], dtype=object)




```python
df.shape
```




    (5, 3)




```python
df.mean() 
```




    col2    7.00
    col3    3.56
    dtype: float64




```python
df1 = pd.DataFrame({'A':[1,2,3]},index=[1,2,3])
df2 = pd.DataFrame({'A':[1,2,3]},index=[3,1,2])
df1-df2
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
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(index='五',columns='col1')
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
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>一</th>
      <td>5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>二</th>
      <td>6</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>三</th>
      <td>7</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>四</th>
      <td>8</td>
      <td>4.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['col1']=[1,2,3,4,5]
del df['col1']
df
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
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>一</th>
      <td>5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>二</th>
      <td>6</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>三</th>
      <td>7</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>四</th>
      <td>8</td>
      <td>4.6</td>
    </tr>
    <tr>
      <th>五</th>
      <td>9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['col1']=[1,2,3,4,5]
df.pop('col1')
```




    一    1
    二    2
    三    3
    四    4
    五    5
    Name: col1, dtype: int64




```python
df
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
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>一</th>
      <td>5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>二</th>
      <td>6</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>三</th>
      <td>7</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>四</th>
      <td>8</td>
      <td>4.6</td>
    </tr>
    <tr>
      <th>五</th>
      <td>9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1['B']=list('abc')
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
      <th>1</th>
      <td>1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.assign(C=pd.Series(list('def')))
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>a</td>
      <td>e</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>b</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>c</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
      <th>1</th>
      <td>1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.select_dtypes(include=['number']).head()
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
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>一</th>
      <td>5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>二</th>
      <td>6</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>三</th>
      <td>7</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>四</th>
      <td>8</td>
      <td>4.6</td>
    </tr>
    <tr>
      <th>五</th>
      <td>9</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.select_dtypes(include=['float']).head()
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
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>一</th>
      <td>1.3</td>
    </tr>
    <tr>
      <th>二</th>
      <td>2.5</td>
    </tr>
    <tr>
      <th>三</th>
      <td>3.6</td>
    </tr>
    <tr>
      <th>四</th>
      <td>4.6</td>
    </tr>
    <tr>
      <th>五</th>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
s = df.mean()
s.name='to_DataFrame'
s
```




    col2    7.00
    col3    3.56
    Name: to_DataFrame, dtype: float64




```python
s.to_frame()
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
      <th>to_DataFrame</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>col2</th>
      <td>7.00</td>
    </tr>
    <tr>
      <th>col3</th>
      <td>3.56</td>
    </tr>
  </tbody>
</table>
</div>




```python
s.to_frame().T
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
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>to_DataFrame</th>
      <td>7.0</td>
      <td>3.56</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv('Documents/Pandas教程/joyful-pandas-master/data/table.csv')
```


```python
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
df.tail()
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
      <th>30</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>2401</td>
      <td>F</td>
      <td>street_2</td>
      <td>192</td>
      <td>62</td>
      <td>45.3</td>
      <td>A</td>
    </tr>
    <tr>
      <th>31</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>2402</td>
      <td>M</td>
      <td>street_7</td>
      <td>166</td>
      <td>82</td>
      <td>48.7</td>
      <td>B</td>
    </tr>
    <tr>
      <th>32</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>2403</td>
      <td>F</td>
      <td>street_6</td>
      <td>158</td>
      <td>60</td>
      <td>59.7</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>33</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>2404</td>
      <td>F</td>
      <td>street_2</td>
      <td>160</td>
      <td>84</td>
      <td>67.7</td>
      <td>B</td>
    </tr>
    <tr>
      <th>34</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>2405</td>
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
df.head(3)
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
  </tbody>
</table>
</div>




```python
df['Physics'].nunique()
```




    7




```python
df['Physics'].unique()
```




    array(['A+', 'B+', 'B-', 'A-', 'B', 'A', 'C'], dtype=object)




```python
df['Physics'].count()
```




    35




```python

df['Physics'].value_counts()
```




    B+    9
    B     8
    B-    6
    A     4
    A-    3
    A+    3
    C     2
    Name: Physics, dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 35 entries, 0 to 34
    Data columns (total 9 columns):
    School     35 non-null object
    Class      35 non-null object
    ID         35 non-null int64
    Gender     35 non-null object
    Address    35 non-null object
    Height     35 non-null int64
    Weight     35 non-null int64
    Math       35 non-null float64
    Physics    35 non-null object
    dtypes: float64(1), int64(3), object(5)
    memory usage: 2.6+ KB



```python
df.describe()
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
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>35.00000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1803.00000</td>
      <td>174.142857</td>
      <td>74.657143</td>
      <td>61.351429</td>
    </tr>
    <tr>
      <th>std</th>
      <td>536.87741</td>
      <td>13.541098</td>
      <td>12.895377</td>
      <td>19.915164</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1101.00000</td>
      <td>155.000000</td>
      <td>53.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1204.50000</td>
      <td>161.000000</td>
      <td>63.000000</td>
      <td>47.400000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2103.00000</td>
      <td>173.000000</td>
      <td>74.000000</td>
      <td>61.700000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2301.50000</td>
      <td>187.500000</td>
      <td>82.000000</td>
      <td>77.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2405.00000</td>
      <td>195.000000</td>
      <td>100.000000</td>
      <td>97.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(percentiles=[.05, .25, .75, .95])
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
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>35.00000</td>
      <td>35.000000</td>
      <td>35.000000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1803.00000</td>
      <td>174.142857</td>
      <td>74.657143</td>
      <td>61.351429</td>
    </tr>
    <tr>
      <th>std</th>
      <td>536.87741</td>
      <td>13.541098</td>
      <td>12.895377</td>
      <td>19.915164</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1101.00000</td>
      <td>155.000000</td>
      <td>53.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>5%</th>
      <td>1102.70000</td>
      <td>157.000000</td>
      <td>56.100000</td>
      <td>32.640000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1204.50000</td>
      <td>161.000000</td>
      <td>63.000000</td>
      <td>47.400000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2103.00000</td>
      <td>173.000000</td>
      <td>74.000000</td>
      <td>61.700000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2301.50000</td>
      <td>187.500000</td>
      <td>82.000000</td>
      <td>77.100000</td>
    </tr>
    <tr>
      <th>95%</th>
      <td>2403.30000</td>
      <td>193.300000</td>
      <td>97.600000</td>
      <td>90.040000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2405.00000</td>
      <td>195.000000</td>
      <td>100.000000</td>
      <td>97.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Physics'].describe()
```




    count     35
    unique     7
    top       B+
    freq       9
    Name: Physics, dtype: object




```python
df['Math'].idxmax()
```




    5




```python
df['Math'].nlargest(3)
```




    5     97.0
    28    95.5
    11    87.7
    Name: Math, dtype: float64




```python
df['Math'].head()
```




    0    34.0
    1    32.5
    2    87.2
    3    80.4
    4    84.8
    Name: Math, dtype: float64




```python
df['Math'].clip(33,80).head()
```




    0    34.0
    1    33.0
    2    80.0
    3    80.0
    4    80.0
    Name: Math, dtype: float64




```python
df['Math'].mad()
```




    16.924244897959188




```python
df['Address'].head()
```




    0    street_1
    1    street_2
    2    street_2
    3    street_2
    4    street_4
    Name: Address, dtype: object




```python
df['Address'].replace(['street_1','street_2'],['one','two']).head()
```




    0         one
    1         two
    2         two
    3         two
    4    street_4
    Name: Address, dtype: object




```python
df.replace({'Address':{'street_1':'one','street_2':'two'}}).head()
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
      <td>one</td>
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
      <td>two</td>
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
      <td>two</td>
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
      <td>two</td>
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
df['Math'].apply(lambda x:str(x)+'!').head() 
```




    0    34.0!
    1    32.5!
    2    87.2!
    3    80.4!
    4    84.8!
    Name: Math, dtype: object




```python
df.apply(lambda x:x.apply(lambda x:str(x)+'!')).head()
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
      <td>S_1!</td>
      <td>C_1!</td>
      <td>1101!</td>
      <td>M!</td>
      <td>street_1!</td>
      <td>173!</td>
      <td>63!</td>
      <td>34.0!</td>
      <td>A+!</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S_1!</td>
      <td>C_1!</td>
      <td>1102!</td>
      <td>F!</td>
      <td>street_2!</td>
      <td>192!</td>
      <td>73!</td>
      <td>32.5!</td>
      <td>B+!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S_1!</td>
      <td>C_1!</td>
      <td>1103!</td>
      <td>M!</td>
      <td>street_2!</td>
      <td>186!</td>
      <td>82!</td>
      <td>87.2!</td>
      <td>B+!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S_1!</td>
      <td>C_1!</td>
      <td>1104!</td>
      <td>F!</td>
      <td>street_2!</td>
      <td>167!</td>
      <td>81!</td>
      <td>80.4!</td>
      <td>B-!</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S_1!</td>
      <td>C_1!</td>
      <td>1105!</td>
      <td>F!</td>
      <td>street_4!</td>
      <td>159!</td>
      <td>64!</td>
      <td>84.8!</td>
      <td>B+!</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index('Math').head()
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
      <th>Physics</th>
    </tr>
    <tr>
      <th>Math</th>
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
      <th>34.0</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1101</td>
      <td>M</td>
      <td>street_1</td>
      <td>173</td>
      <td>63</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>32.5</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1102</td>
      <td>F</td>
      <td>street_2</td>
      <td>192</td>
      <td>73</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>87.2</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1103</td>
      <td>M</td>
      <td>street_2</td>
      <td>186</td>
      <td>82</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>80.4</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1104</td>
      <td>F</td>
      <td>street_2</td>
      <td>167</td>
      <td>81</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>84.8</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1105</td>
      <td>F</td>
      <td>street_4</td>
      <td>159</td>
      <td>64</td>
      <td>B+</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index('Math').sort_index().head()
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
      <th>Physics</th>
    </tr>
    <tr>
      <th>Math</th>
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
      <th>31.5</th>
      <td>S_1</td>
      <td>C_3</td>
      <td>1301</td>
      <td>M</td>
      <td>street_4</td>
      <td>161</td>
      <td>68</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>32.5</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1102</td>
      <td>F</td>
      <td>street_2</td>
      <td>192</td>
      <td>73</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>32.7</th>
      <td>S_2</td>
      <td>C_3</td>
      <td>2302</td>
      <td>M</td>
      <td>street_5</td>
      <td>171</td>
      <td>88</td>
      <td>A</td>
    </tr>
    <tr>
      <th>33.8</th>
      <td>S_1</td>
      <td>C_2</td>
      <td>1204</td>
      <td>F</td>
      <td>street_5</td>
      <td>162</td>
      <td>63</td>
      <td>B</td>
    </tr>
    <tr>
      <th>34.0</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1101</td>
      <td>M</td>
      <td>street_1</td>
      <td>173</td>
      <td>63</td>
      <td>A+</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_values(by='Class').head()
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
      <th>19</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>2105</td>
      <td>M</td>
      <td>street_4</td>
      <td>170</td>
      <td>81</td>
      <td>34.2</td>
      <td>A</td>
    </tr>
    <tr>
      <th>18</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>2104</td>
      <td>F</td>
      <td>street_5</td>
      <td>159</td>
      <td>97</td>
      <td>72.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>16</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>2102</td>
      <td>F</td>
      <td>street_6</td>
      <td>161</td>
      <td>61</td>
      <td>50.6</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>15</th>
      <td>S_2</td>
      <td>C_1</td>
      <td>2101</td>
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
df.sort_values(by=['Address','Height']).head()
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
      <th>11</th>
      <td>S_1</td>
      <td>C_3</td>
      <td>1302</td>
      <td>F</td>
      <td>street_1</td>
      <td>175</td>
      <td>57</td>
      <td>87.7</td>
      <td>A-</td>
    </tr>
    <tr>
      <th>23</th>
      <td>S_2</td>
      <td>C_2</td>
      <td>2204</td>
      <td>M</td>
      <td>street_1</td>
      <td>175</td>
      <td>74</td>
      <td>47.2</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>33</th>
      <td>S_2</td>
      <td>C_4</td>
      <td>2404</td>
      <td>F</td>
      <td>street_2</td>
      <td>160</td>
      <td>84</td>
      <td>67.7</td>
      <td>B</td>
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
  </tbody>
</table>
</div>



## 问题
### 【问题一】 Series和DataFrame有哪些常见属性和方法？
对于一个Series，其中最常用的属性为值（values），索引（index），名字（name），类型（dtype）。
对于DataFrame，属性有index、columns、values、shape。
方法是mean（）。
### 【问题二】 value_counts会统计缺失值吗？
不会
### 【问题三】 与idxmax和nlargest功能相反的是哪两组函数？
idxmin、nsmallest
### 【问题四】 在常用函数一节中，由于一些函数的功能比较简单，因此没有列入，现在将它们列在下面，请分别说明它们的用途并尝试使用。
sum/mean/median/mad/min/max/abs/std/var/quantile/cummax/cumsum/cumprod
### 【问题五】 df.mean(axis=1)是什么意思？它与df.mean()的结果一样吗？第一问提到的函数也有axis参数吗？怎么使用？



## 练习
1.【练习一】 现有一份关于美剧《权力的游戏》剧本的数据集，请解决以下问题：¶

（a）在所有的数据中，一共出现了多少人物？

（b）以单元格计数（即简单把一个单元格视作一句），谁说了最多的话？

（c）以单词计数，谁说了最多的单词

2.【练习二】现有一份关于科比的投篮数据集，请解决如下问题：

（a）哪种action_type和combined_shot_type的组合是最多的？

（b）在所有被记录的game_id中，遭遇到最多的opponent是一个支？


```python
#练习1-a
df = pd.read_csv('Documents/Pandas教程/joyful-pandas-master/data/Game_of_Thrones_Script.csv')
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
      <th>Release Date</th>
      <th>Season</th>
      <th>Episode</th>
      <th>Episode Title</th>
      <th>Name</th>
      <th>Sentence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011/4/17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>waymar royce</td>
      <td>What do you expect? They're savages. One lot s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011/4/17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>will</td>
      <td>I've never seen wildlings do a thing like this...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011/4/17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>waymar royce</td>
      <td>How close did you get?</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011/4/17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>will</td>
      <td>Close as any man would.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011/4/17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>gared</td>
      <td>We should head back to the wall.</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Name'].nunique()
```




    564




```python
#练习1-b
df['Name'].value_counts().index[0]
```




    'tyrion lannister'




```python
#练习1-c
df_words = df.assign(Words=df['Sentence'].apply(lambda x:len(x.split()))).sort_values(by='Name')
df_words.head()
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
      <th>Release Date</th>
      <th>Season</th>
      <th>Episode</th>
      <th>Episode Title</th>
      <th>Name</th>
      <th>Sentence</th>
      <th>Words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>276</th>
      <td>2011/4/17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>a voice</td>
      <td>It's Maester Luwin, my lord.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3012</th>
      <td>2011/6/19</td>
      <td>Season 1</td>
      <td>Episode 10</td>
      <td>Fire and Blood</td>
      <td>addam marbrand</td>
      <td>ls it true about Stannis and Renly?</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3017</th>
      <td>2011/6/19</td>
      <td>Season 1</td>
      <td>Episode 10</td>
      <td>Fire and Blood</td>
      <td>addam marbrand</td>
      <td>Kevan Lannister</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13610</th>
      <td>2014/6/8</td>
      <td>Season 4</td>
      <td>Episode 9</td>
      <td>The Watchers on the Wall</td>
      <td>aemon</td>
      <td>And what is it that couldn't wait until mornin...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>13614</th>
      <td>2014/6/8</td>
      <td>Season 4</td>
      <td>Episode 9</td>
      <td>The Watchers on the Wall</td>
      <td>aemon</td>
      <td>Oh, no need. I know my way around this library...</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>




```python
L_count = []
N_words = list(zip(df_words['Name'],df_words['Words']))
for i in N_words:
    if i == N_words[0]:
        L_count.append(i[1])
        last = i[0]
    else:
        L_count.append(L_count[-1]+i[1] if i[0]==last else i[1])
        last = i[0]
df_words['Count']=L_count
df_words['Name'][df_words['Count'].idxmax()]
```




    'tyrion lannister'




```python
#练习2-a
df = pd.read_csv('Documents/Pandas教程/joyful-pandas-master/data/Kobe_data.csv',index_col='shot_id')
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
      <th>action_type</th>
      <th>combined_shot_type</th>
      <th>game_event_id</th>
      <th>game_id</th>
      <th>lat</th>
      <th>loc_x</th>
      <th>loc_y</th>
      <th>lon</th>
      <th>minutes_remaining</th>
      <th>period</th>
      <th>...</th>
      <th>shot_made_flag</th>
      <th>shot_type</th>
      <th>shot_zone_area</th>
      <th>shot_zone_basic</th>
      <th>shot_zone_range</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>game_date</th>
      <th>matchup</th>
      <th>opponent</th>
    </tr>
    <tr>
      <th>shot_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>1</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>10</td>
      <td>20000012</td>
      <td>33.9723</td>
      <td>167</td>
      <td>72</td>
      <td>-118.1028</td>
      <td>10</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>2PT Field Goal</td>
      <td>Right Side(R)</td>
      <td>Mid-Range</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000/10/31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>12</td>
      <td>20000012</td>
      <td>34.0443</td>
      <td>-157</td>
      <td>0</td>
      <td>-118.4268</td>
      <td>10</td>
      <td>1</td>
      <td>...</td>
      <td>0.0</td>
      <td>2PT Field Goal</td>
      <td>Left Side(L)</td>
      <td>Mid-Range</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000/10/31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>35</td>
      <td>20000012</td>
      <td>33.9093</td>
      <td>-101</td>
      <td>135</td>
      <td>-118.3708</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>1.0</td>
      <td>2PT Field Goal</td>
      <td>Left Side Center(LC)</td>
      <td>Mid-Range</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000/10/31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>43</td>
      <td>20000012</td>
      <td>33.8693</td>
      <td>138</td>
      <td>175</td>
      <td>-118.1318</td>
      <td>6</td>
      <td>1</td>
      <td>...</td>
      <td>0.0</td>
      <td>2PT Field Goal</td>
      <td>Right Side Center(RC)</td>
      <td>Mid-Range</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000/10/31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Driving Dunk Shot</td>
      <td>Dunk</td>
      <td>155</td>
      <td>20000012</td>
      <td>34.0443</td>
      <td>0</td>
      <td>0</td>
      <td>-118.2698</td>
      <td>6</td>
      <td>2</td>
      <td>...</td>
      <td>1.0</td>
      <td>2PT Field Goal</td>
      <td>Center(C)</td>
      <td>Restricted Area</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000/10/31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
pd.Series(list(zip(df['action_type'],df['combined_shot_type']))).value_counts().index[0]
```




    ('Jump Shot', 'Jump Shot')




```python
#练习2-b
pd.Series(list(list(zip(*(pd.Series(list(zip(df['game_id'],df['opponent'])))
                          .unique()).tolist()))[1])).value_counts().index[0]
```




    'SAS'


