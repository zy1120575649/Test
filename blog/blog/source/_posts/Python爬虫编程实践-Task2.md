---
title: Python爬虫编程实践-Task2
date: 2020-04-23 19:21:23
tags:
    - DataWhale
    - Python爬虫
categories: Python爬虫编程实践
---

## Beautiful Soup库入门
1. 学习beautifulsoup基础知识。

2. 使用beautifulsoup解析HTML页面。

Beautiful Soup 是一个HTML/XML 的解析器，主要用于解析和提取 HTML/XML 数据。 
它基于HTML DOM 的，会载入整个文档，解析整个DOM树，因此时间和内存开销都会大很多，所以性能要低于lxml。

BeautifulSoup 用来解析 HTML 比较简单，API非常人性化，支持CSS选择器、Python标准库中的HTML解析器，也支持 lxml 的 XML解析器。

虽然说BeautifulSoup4 简单容易比较上手，但是匹配效率还是远远不如正则以及xpath的，一般不推荐使用，推荐正则的使用。

第一步：pip install beautifulsoup4 ，万事开头难，先安装 beautifulsoup4，安装成功后就完成了第一步。

第二步：导入from bs4 import BeautifulSoup

第三步：创建 Beautiful Soup对象   soup = BeautifulSoup(html，'html.parser') 

## 学习xpath
1.  学习目标：
学习xpath，使用lxml+xpath提取内容。

使用xpath提取丁香园论坛的回复内容。

抓取丁香园网页：http://www.dxy.cn/bbs/thread/626626#626626 。
2. Xpath常用的路径表达式：
XPath即为XML路径语言（XML Path Language），它是一种用来确定XML文档中某部分位置的语言。
在XPath中，有七种类型的节点：元素、属性、文本、命名空间、处理指令、注释以及文档（根）节点。
XML文档是被作为节点树来对待的。
XPath使用路径表达式在XML文档中选取节点。节点是通过沿着路径选取的。

3. 使用lxml解析
4. 实战：爬取丁香园-用户名和回复内容


```python
# 导入库
from lxml import etree
import requests

url = "http://www.dxy.cn/bbs/thread/626626#626626"
```


```python
req = requests.get(url)
html = req.text
# html
```


```python
tree = etree.HTML(html) 
tree
```




    <Element html at 0x110f226e0>




```python
user = tree.xpath('')
# print(user)
content = tree.xpath('')
```


```python
results = []
for i in range(0, len(user)):
    # print(user[i].strip()+":"+content[i].xpath('string(.)').strip())
    # print("*"*80)
    # 因为回复内容中有换行等标签，所以需要用string()来获取数据
    results.append(user[i].strip() + ":  " + content[i].xpath('string(.)').strip())
```


```python
# 打印爬取的结果
for i,result in zip(range(0, len(user)),results):
    print("user"+ str(i+1) + "-" + result)
    print("*"*100)
```

## 学习正则表达式 re
re库的主要功能函数：

re.search() 在一个字符串中搜索匹配正则表达式的第一个位置，返回match对象
re.search(pattern, string, flags=0)

re.match() 从一个字符串的开始位置起匹配正则表达式，返回match对象
re.match(pattern, string, flags=0)

re.findall() 搜索字符串，以列表类型返回全部能匹配的子串
re.findall(pattern, string, flags=0)

re.split() 将一个字符串按照正则表达式匹配结果进行分割，返回列表类型
re.split(pattern, string, maxsplit=0, flags=0)

re.finditer() 搜索字符串，返回一个匹配结果的迭代类型，每个迭代元素是match对象
re.finditer(pattern, string, flags=0)

re.sub() 在一个字符串中替换所有匹配正则表达式的子串，返回替换后的字符串


