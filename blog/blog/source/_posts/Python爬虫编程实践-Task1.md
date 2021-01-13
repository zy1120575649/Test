---
title: Python爬虫编程实践-Task1
date: 2020-04-21 10:54:09
tags:
    - DataWhale
    - Python爬虫
categories: Python爬虫编程实践
---

HTTP是一个客户端（用户）和服务器端（网站）之间进行请求和应答的标准。通过使用网页浏览器、网络爬虫或者其他工具，客户端可以向服务器上的指定端口（默认端口为80）发起一个HTTP请求。这个客户端成为客户代理（user agent）。应答服务器上存储着一些资源码，比如HTML文件和图像。这个应答服务器成为源服务器（origin server）。在用户代理和源服务器中间可能存在多个“中间层”，比如代理服务器、网关或者隧道（tunnel）。尽管TCP/IP是互联网最流行的协议，但HTTP中并没有规定必须使用它或它支持的层。
<!--more-->

事实上。HTTP可以在互联网协议或其他网络上实现。HTTP假定其下层协议能够提供可靠的传输，因此，任何能够提供这种保证的协议都可以使用。使用TCP/IP协议族时RCP作为传输层。通常由HTTP客户端发起一个请求，创建一个到服务器指定端口（默认是80端口）的TCP链接。HTTP服务器则在该端口监听客户端的请求。一旦收到请求，服务器会向客户端返回一个状态（比如“THTTP/1.1 200 OK”），以及请求的文件、错误信息等响应内容。

HTTP的请求方法有很多种，主要包括以下几个：

1. GET：向指定的资源发出“显示”请求。GET方法应该只用于读取数据，而不应当被用于“副作用”的操作中（例如在Web Application中）。其中一个原因是GET可能会被网络蜘蛛等随意访问。

2. HEAD：与GET方法一样，都是向服务器发出直顶资源的请求，只不过服务器将不会出传回资源的内容部分。它的好处在于，使用这个方法可以在不必传输内容的情况下，将获取到其中“关于该资源的信息”（元信息或元数据）。

3. POST：向指定资源提交数据，请求服务器进行处理（例如提交表单或者上传文件）。数据被包含在请求文本中。这个请求可能会创建新的资源或修改现有资源，或二者皆有。

4. PUT：向指定资源位置上传输最新内容。

5. DELETE：请求服务器删除Request-URL所标识的资源，或二者皆有。

6. TRACE：回显服务器收到的请求，主要用于测试或诊断。

7. OPTIONS：这个方法可使服务器传回该资源所支持的所有HTTP请求方法。用“*”来代表资源名称向Web服务器发送OPTIONS请求，可以测试服务器共能是否正常。

8. CONNECT：HTTP/1.1 协议中预留给能够将连接改为管道方式的代理服务器。通常用于SSL加密服务器的连接（经由非加密的HTTP代理服务器）。方法名称是区分大小写的。当某个请求所针对的资源不支持对应的请求方法的时候，服务器应当返回状态码405（Method Not Allowed），当服务器不认识或者不支持对应的请求方法的时候，应当返回状态码501（Not Implemented）。


html和css都会跳过。

Chrome的开发者模式为用户提供了下面几组工具。

1. Elements：允许用户从浏览器的角度来观察网页，用户可以借此看到Chrome渲染页面所需要的HTML、CSS和DOM（Document Object Model）对象。

2. Network：可以看到网页向服务气请求了哪些资源、资源的大小以及加载资源的相关信息。此外，还可以查看HTTP的请求头、返回内容等。

3. Source：即源代码面板，主要用来调试JavaScript。

4. Console：即控制台面板，可以显示各种警告与错误信息。在开发期间，可以使用控制台面板记录诊断信息，或者使用它作为shell在页面上与JavaScript交互。

5. Performance：使用这个模块可以记录和查看网站生命周期内发生的各种事情来提高页面运行时的性能。

6. Memory：这个面板可以提供比Performance更多的信息，比如跟踪内存泄漏。

7. Application：检查加载的所有资源。

8. Security：即安全面板，可以用来处理证书问题等。


一个网络爬虫程序最普遍的过程：

1. 访问站点；
2. 定位所需的信息；
3. 得到并处理信息。


```python
import requests
url = 'https://www.python.org/dev/peps/pep-0020/'
res = requests.get(url)
text = res.text
```


```python
## 爬取python之禅并存入txt文件

with open('zon_of_python.txt', 'w') as f:
    f.write(text[text.find('<pre')+28:text.find('</pre>')-1])
print(text[text.find('<pre')+28:text.find('</pre>')-1])
```


```python
import urllib
url = 'https://www.python.org/dev/peps/pep-0020/'
res = urllib.request.urlopen(url).read().decode('utf-8')
print(res[res.find('<pre')+28:res.find('</pre>')-1])
```

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!



使用API：

所谓的采集网络数据，并不一定必须从网页中抓取数据，而api（Application Programming Iterface）的用处就在这里：API为开发者提供了方便友好的接口，不同的开发者用不同的语言都能获取相同的数据。目前API一般会以XML（Extensible Markup Language，可拓展标记语言）或者JSON（JavaScript Object Notation）格式来返回服务器响应，其中JSON数据格式越来越受到人们的欢迎，我们后面的课程也会详细介绍JSON格式。
