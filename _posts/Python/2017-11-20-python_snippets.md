---
layout: post
title: Python 有意思的几个snippets
category: python
tags: [python]
description: snippets
---

整理几个有意思的Python snippets. 

<!-- more -->

### 目录
{:.no_toc}

* 目录
{:toc}

### 开头结尾

``` python
#!/usr/bin/env python  
# -*- coding: utf-8 -*-
import sys  
reload(sys) 
sys.setdefaultencoding('utf8')  

# 结尾
if __name__ == '__main__':
    main()

```


### 函数超时设置
``` python
import time, signal


def timeout(timeout=1):
    class TimeoutError(Exception):
        pass
    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout)
    for i in range(1000):
        print("deal %s " %i)
        time.sleep(0.5)

# 打印到1的时候, 会退出, 理由是TimeoutError
try:
    timeout()
except:
    print("finish")
```


### 单元测试
这个是写代码必备的, 用的多.
``` python
import unittest


class MyTest(unittest.TestCase):
    def setUp(self):
        # 设置一些共享的初始化参数
        self.x = 1

    def tearDown(self):
        pass

    def test_ua(self):
        y = 2 * self.x + 1
        self.assertEquals(y, 3)
    
unittest.main()
```

### 时间处理函数
工作里面因为经常有unix时间和字符串时间的转换,所以写了一个脚本去转换. 
另外遇到时间处理, 总是忘记一些函数, 把部分核心代码放进来. 

``` python
import time, datetime

# 字符串转时间
day = "20160123"
day_item = time.strptime(day, '%Y%m%d')
day_unix = time.mktime(day_item)
one_day_minus = day_unix - 86400 * 1

#时间转unix
d = datetime.datetime(2015,12,10)
unixtime = time.mktime(d.timetuple())

# unix时间转字符串
print time.strftime("%Y-%m-%d %H:%M", time.localtime(1453520966))
print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

# 日期加减
start_date = datetime.datetime(2015,12,18)
a = start_date.strftime("%Y%m%d%H%M")
if a == "201512310000":
    start_date += datetime.timedelta(days=1)
```

#### 并发处理函数
``` python
from multiprocessing import Process, Queue, Pool

# 多线程处理
class FuncRunner(object):

    def __init__(self, func, cache_size=80, thread_num=3):
        '''
        并发跑任务,输入和输出的顺序未必是一致的, 需要手动确认
        func: 函数引用
        '''
        self.cache_size = cache_size
        self.push_queue = Queue(maxsize=cache_size)
        self.res_queue = Queue()
        self.func = func
        self.threads = []
        for i in range(thread_num):
            p = Process(target=self.__deal_queue)
            p.daemon = True
            p.start()
            self.threads.append(p)

    def run(self, params):
        '''
        params: must be a list of param
        '''
        for param in params:
            self.push_queue.put(param)
        result = []
        while len(result) < len(params):
            result.append(self.res_queue.get())

        for threads in self.threads:
            threads.terminate()
        return result

    def __deal_queue(self):
        while True:
            param = self.push_queue.get()
            self.res_queue.put(self.func(param))


def f(x):
    return x ** 2

# 进程池处理, 这个函数不一定实用, 官网例子比较多
def pool_deal(data, f):
    lines = []
    pool = Pool(processes=10)
    for i in data:
        line = pool.apply_async(f, (i, ))
        lines.append(line)
    pool.close()
    pool.join()
    result = []
    for line in lines:
        r = line.get() if line.successful() else None
        result.append(r)

    return result


def test():
    nums = xrange(200)
    runner = FuncRunner(f)
    res1 = runner.run(nums)
    res2 = pool_deal(nums, f)
    print res1, res2

```


#### 神奇小函数
``` python

# eval, 字符串格式互转
In [10]: data  #{1: 2, 2: 3}
In [11]: str(data) # '{1: 2, 2: 3}'
In [12]: d = eval(str(data))
Out[13]: {1: 2, 2: 3}

# 路径遍历
os.walk(rootdir)

#将x转换为一个整数
int(x [,base ])
#创建一个复数
complex(real [,imag ])
# 将对象 x 转换为表达式字符串
repr(x)
# 将一个整数转换为一个字符
chr(x)
# 将一个整数转换为Unicode字符
unichr(x)
# 将一个字符转换为它的整数值
ord(x)
# 将一个整数转换为一个十六进制字符串
hex(x)
# 将一个整数转换为一个八进制字符串
oct(x)   

``` 