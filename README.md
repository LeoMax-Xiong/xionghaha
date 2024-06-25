# xionghaha

pip
环境配置

百度源配置

配置pip源，将以下内容写入pip.conf文件中，普通用户路径为~/.pip/pip.conf（自己创建目录），root用户路径为/etc/pip.conf
``` shell
[global]
timeout = 60
index = https://pip.baidu-int.com/search/
index-url = https://pip.baidu-int.com/simple/
trusted-host = pip.baidu-int.com
[list]
format = columns
```
镜像源
百度：http://pip.baidu-int.com/simple/
https://mirror.baidu.com/pypi/simple/
阿里：https://mirrors.aliyun.com/pypi/simple/
清华：https://pypi.tuna.tsinghua.edu.cn/simple/
豆瓣：http://pypi.douban.com/simple/
