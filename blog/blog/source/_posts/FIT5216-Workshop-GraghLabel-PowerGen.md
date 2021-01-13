---
title: FIT5216 Workshop-GraghLabel&PowerGen
date: 2020-03-28 11:11:50
tags:
	-FIT5216 Week2
categories: FIT5216 Modelling Discrete Optimization Problems
---

这两个题目是在第二周的workshop上讨论做的，第一道简单，第二道有点难，难在miniizinc的语法，最好记住。

<!--more-->

# GraghLabel

## Problem Statement

![image-20200328111720203](/image-20200328111720203.png)

![image-20200328111736614](/image-20200328111736614.png)

## Answers

```python
var 1..8:a;
var 1..8:b;
var 1..8:c;
var 1..8:d;
var 1..8:e;
var 1..8:f;
var 1..8:g;
var 1..8:h;

constraint abs(a-b)>1;
constraint abs(a-c)>1;
constraint abs(a-d)>1;
constraint abs(b-e)>1;
constraint abs(b-f)>1;
constraint abs(b-c)>1;
constraint abs(c-e)>1;
constraint abs(c-f)>1;
constraint abs(c-d)>1;
constraint abs(c-g)>1;
constraint abs(d-f)>1;
constraint abs(d-g)>1;
constraint abs(e-h)>1;
constraint abs(e-f)>1;
constraint abs(f-h)>1;
constraint abs(f-g)>1;
constraint abs(g-h)>1;
include "alldifferent.mzn";
constraint alldifferent([a,b,c,d,e,f,g,h]);
```



# PowerGen

## Problem Statement

![image-20200328111858984](/image-20200328111858984.png)

![image-20200328111916857](/image-20200328111916857.png)

## Answers

```python
% power generation
int: T;                 % decades
array[1..T] of int: e;  % expected requirements
array[1..T] of int: a;  % current production

array[1..T] of var 0..infinity: N; % number of nuclear power plants built each decade
array[1..T] of var 0..infinity: C; % number of coal power plants built each decade
array[1..T] of var 0..infinity: S; % number of solar power plants built each decade

var 0..infinity: cost;  % costs of building all new power plants

constraint forall ( i in 1..T ) (
   sum ( j in max(1,i-5)..i ) ( N[j] * 4 ) + sum ( j in max(1,i-1)..i ) ( C[j] ) + sum ( j in max(1,i-2)..i ) ( S[j] ) >= e[i] - a[i]
);

constraint forall ( i in 1..T ) (
   sum ( j in max(1,i-5)..i ) ( N[j] * 4 ) <= 0.4 * ( a[i] + sum ( j in max(1,i-5)..i ) ( N[j] * 4 ) + sum ( j in max(1,i-1)..i ) ( C[j] ) + sum ( j in max(1,i-2)..i ) ( S[j] ) )
);

constraint forall ( i in 1..T ) (
   sum ( j in max(1,i-2)..i ) ( S[j] ) >= 0.2 * ( a[i] + sum ( j in max(1,i-5)..i ) ( N[j] * 4 ) + sum ( j in max(1,i-1)..i ) ( C[j] ) + sum ( j in max(1,i-2)..i ) ( S[j] ) )
);

constraint C[1] + sum ( i in 2..T ) ( C[i] + C[i-1] ) <= 10;

cost = sum ( i in 1..T ) ( N[i] * 10 + C[i] + S[i] * 2 );

solve minimize cost;
                           

```