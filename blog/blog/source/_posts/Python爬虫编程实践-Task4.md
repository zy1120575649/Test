---
title: Python爬虫编程实践-Task4
date: 2020-04-26 19:24:38
tags:
    - DataWhale
    - Python爬虫
categories: Python爬虫编程实践
---

1. 了解ajax加载
2. 通过chrome的开发者工具，监控网络请求，并分析
3. 用selenium完成爬虫


```python
import time
from  selenium import webdriver
driver=webdriver.Chrome(executable_path="D:\chromedriver\chromedriver.exe")
driver.get("https://news.qq.com")
#了解ajax加载
for i in range(1,100):
    time.sleep(2)
    driver.execute_script("window.scrollTo(window.scrollX, %d);"%(i*200))
```


```python
from bs4 import BeautifulSoup
html=driver.page_source
bsObj=BeautifulSoup(html,"lxml")
```


```python
jxtits=bsObj.find_all("div",{"class":"jx-tit"})[0].find_next_sibling().find_all("li")
```


```python
print("index",",","title",",","url")
for i,jxtit in enumerate(jxtits):
#     print(jxtit)
    
    try:
        text=jxtit.find_all("img")[0]["alt"]
    except:
        text=jxtit.find_all("div",{"class":"lazyload-placeholder"})[0].text
    try:
        url=jxtit.find_all("a")[0]["href"]
    except:
        print(jxtit)
    print(i+1,",",text,",",url) 
```

进阶加餐-知乎爬虫

链接如下
https://www.zhihu.com/search?q=Datawhale&utm_content=search_history&type=content
用requests库实现，不能用selenium网页自动化

提示：
该链接需要登录，可通过github等，搜索知乎登录的代码实现，并理解其中的逻辑，此任务允许复制粘贴代码
与上面ajax加载类似，这次的ajax加载需要用requests完成爬取，最终存储样式随意，但是通过Chrome的开发者工具，分析出ajax的流程需要写出来


```python
import requests
from http import cookiejar
Session=requests.session()
Session.cookies = cookiejar.LWPCookieJar(filename='./cookies.txt')
Session.cookies.load(ignore_discard=True)
Session.headers={
            'Host': 'www.zhihu.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
        }
r=Session.get("https://www.zhihu.com/search?q=Datawhale&utm_content=search_history&type=content")
r.encoding="utf-8"
```


```python
from bs4 import BeautifulSoup
import re
compiler=re.compile('"next":"(https:\\\\u002F\\\\u002Fapi.zhihu.com\\\\u002Fsearch_v3.*?)"')
r.text
```


```python
bsObj=BeautifulSoup(r.text,"lxml")
url=compiler.findall(r.text)[0]
```


```python
from urllib.parse import unquote
url=unquote(url,encoding="utf-8", errors='replace')
url=url.replace("\\u002F","/")
search_hash_id=re.search("search_hash_id=(.*?)&show_all_topics",url).group(1)
search_hash_id
```


```python
offset=20
lc_idx=21
for i in range(5):
    r=Session.get("https://www.zhihu.com/api/v4/search_v3?t=general&q=Datawhale&correction=1&offset={offset}&limit=20&lc_idx={lc_idx}&show_all_topics=0&search_hash_id={search_hash_id}&vertical_info=0%2C0%2C1%2C0%2C0%2C0%2C0%2C0%2C0%2C0".format(**{"offset":offset+i*20,"lc_idx":lc_idx+i*20,"search_hash_id":search_hash_id}))
    r.encoding="utf-8"
    print(r.json())
    print("\n"*20)
```
