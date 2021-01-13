---
title: Python爬虫编程实践-Task3
date: 2020-04-25 16:53:39
tags:
    - DataWhale
    - Python爬虫
categories: Python爬虫编程实践
---
## session和cookie
Session 是会话的意思，会话是产生在服务端的，用来保存当前用户的会话信息，而 Cookies 是保存在客户端（浏览器），有了 Cookie 以后，客户端（浏览器）再次访问服务端的时候，会将这个 Cookie 带上，这时，服务端可以通过 Cookie 来识别本次请求到底是谁在访问。
<!--more-->
可以简单理解为 Cookies 中保存了登录凭证，我们只要持有这个凭证，就可以在服务端保持一个登录状态。

在爬虫中，有时候遇到需要登录才能访问的网页，只需要在登录后获取了 Cookies ，在下次访问的时候将登录后获取到的 Cookies 放在请求头中，这时，服务端就会认为我们的爬虫是一个正常登录用户。

一个重要概念：

当我们关闭浏览器的时候会自动销毁服务端的会话，这个是错误的，因为在关闭浏览器的时候，浏览器并不会额外的通知服务端说，我要关闭了，你把和我的会话销毁掉吧。

因为服务端的会话是保存在内存中的，虽然一个会话不会很大，但是架不住会话多啊，硬件毕竟是会有限制的，不能无限扩充下去的，所以在服务端设置会话的过期时间就非常有必要。

当然，有没有方式能让浏览器在关闭的时候同步的关闭服务端的会话，当然是可以的，我们可以通过脚本语言 JS 来监听浏览器关闭的动作，当浏览器触发关闭动作的时候，由 JS 像服务端发起一个请求来通知服务端销毁会话。

由于不同的浏览器对 JS 事件的实现机制不一致，不一定保证 JS 能监听到浏览器关闭的动作，所以现在常用的方式还是在服务端自己设置会话的过期时间
## 代理
如何应对IP被封的问题

有几种套路：

1. 修改请求头，模拟浏览器（而不是代码去直接访问）去访问
2. 采用代理IP并轮换
3. 设置访问时间间隔


```python
import requests
import re
import json


def open_proxy_url(url):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
    headers = {'User-Agent': user_agent}
    try:
        r = requests.get(url, headers = headers, timeout = 10)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        print('无法访问网页' + url)


def get_proxy_ip(response):
    proxy_ip_list = []
    soup = BeautifulSoup(response, 'html.parser')
    proxy_ips = soup.find(id = 'ip_list').find_all('tr')
    for proxy_ip in proxy_ips:
        if len(proxy_ip.select('td')) >=8:
            ip = proxy_ip.select('td')[1].text
            port = proxy_ip.select('td')[2].text
            protocol = proxy_ip.select('td')[5].text
            if protocol in ('HTTP','HTTPS','http','https'):
                proxy_ip_list.append(f'{protocol}://{ip}:{port}')
    return proxy_ip_list


def open_url_using_proxy(url, proxy):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
    headers = {'User-Agent': user_agent}
    proxies = {}
    if proxy.startswith(('HTTPS','https')):
        proxies['https'] = proxy
    else:
        proxies['http'] = proxy

    try:
        r = requests.get(url, headers = headers, proxies = proxies, timeout = 10)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return (r.text, r.status_code)
    except:
        print('无法访问网页' + url)
        print('无效代理IP: ' + proxy)
        return False


def check_proxy_avaliability(proxy):
    url = 'http://www.baidu.com'
    result = open_url_using_proxy(url, proxy)
    VALID_PROXY = False
    if result:
        text, status_code = result
        if status_code == 200:
            r_title = re.findall('<title>.*</title>', text)
            if r_title:
                if r_title[0] == '<title>百度一下，你就知道</title>':
                    VALID_PROXY = True
        if VALID_PROXY:
            check_ip_url = 'https://jsonip.com/'
            try:
                text, status_code = open_url_using_proxy(check_ip_url, proxy)
            except:
                return

            print('有效代理IP: ' + proxy)
            with open('valid_proxy_ip.txt','a') as f:
                f.writelines(proxy)
            try:
                source_ip = json.loads(text).get('ip')
                print(f'源IP地址为：{source_ip}')
                print('='*40)
            except:
                print('返回的非json,无法解析')
                print(text)
    else:
        print('无效代理IP: ' + proxy)


if __name__ == '__main__':
    proxy_url = 'https://www.xicidaili.com/'
    proxy_ip_filename = 'proxy_ip.txt'
    text = open(proxy_ip_filename, 'r').read()
    proxy_ip_list = get_proxy_ip(text)
    for proxy in proxy_ip_list:
        check_proxy_avaliability(proxy)
```

## selenium
selenium是什么：一个自动化测试工具（大家都是这么说的）

selenium应用场景：用代码的方式去模拟浏览器操作过程（如：打开浏览器、在输入框里输入文字、回车等），在爬虫方面很有必要

准备工作：

1. 安装selenium（pip install selenium）
2. 安装chromedriver（一个驱动程序，用以启动chrome浏览器，具体的驱动程序需要对应的驱动，在官网上可以找到下载地址） 基本步骤：


```python
from selenium import webdriver  # 启动浏览器需要用到
from selenium.webdriver.common.keys import Keys
```


```python
driver = webdriver.Chrome("chromedriver驱动程序路径")
```


```python
driver.get("http://www.python.org")
```


```python
driver.close()  # 关闭浏览器一个Tab
# or
driver.quit()  # 关闭浏览器窗口
```
