# my_openai_api
在自己的电脑上部署模型兼容openai接口，目前baichuan2-13b-chat-4bits在单张tesla t4显卡就可以跑动，并且效果和速度也还可以，可以和gpt-3.5媲美。
此项目基于flask, transforms单个文件实现openai接口(models, chat, moderations包含流式响应)，可以保证langchain基础调用。
## 最低配置
需要16g显存，如果主机显存不够可以考虑腾讯云的活动，60块钱15天32g内存、t4显卡的主机，非常划算😝，可以跑动baichuan2-13b-chat-4bits。  
  
<a href="https://s2.loli.net/2023/09/25/q7c4jdjocwym1fh.png" target="_blank"><img src="https://s2.loli.net/2023/09/25/q7c4jdjocwym1fh.png" width="60%"></a>  

地址: [https://cloud.tencent.com/act/pro/gpu-study](https://cloud.tencent.com/act/pro/gpu-study)  

如果想要本地运行，t4显卡价格在5600元左右。如果嫌贵，可以考虑2080ti魔改22g版本，某宝2600元左右 🤓️。
## 安装
1. 下载代码
```
git clone https://github.com/billvsme/my_openai_api.git
```
2. 下载baichuan2-13b-chat-4bits模型
```
cd my_openai_api
git clone https://huggingface.co/baichuan-inc/baichuan2-7b-chat-4bits  # 需要先安装好hfs， git lfs install
```
3. 安装venv环境
```
mkdir ~/.venv
python -m venv ~/.venv/ai
. ~/.venv/ai/bin/activate

pip install -r requirements.txt
```
## 启动
```
python my_openai_api.py
或者
gunicorn -b 0.0.0.0:5000 --workers=1  my_openai_api:app
```
## 文档
实现了openai的models, chat, moderations 3个接口  
可以参考https://platform.openai.com/docs/api-reference/chat
```
打开 http://127.0.0.1:5000/apidocs/
```
  
<a href="https://s2.loli.net/2023/09/25/o8i5ge3onfhsaqz.png" target="_blank"><img src="https://s2.loli.net/2023/09/25/o8i5ge3onfhsaqz.png", width="60%" ></a>  

## 使用
替换openai_base_api, 以langchain为例
```
# coding: utf-8
from langchain.llms import openai
from langchain.chat_models import chatopenai
from langchain.callbacks.streaming_stdout import streamingstdoutcallbackhandler
from langchain.schema import (
    humanmessage,
)

openai_api_base = "http://127.0.0.1:5000/v1"
openai_api_key = "test"

# /v1/chat/completions流式响应
chat_model = chatopenai(streaming=true, callbacks=[streamingstdoutcallbackhandler()], openai_api_base=openai_api_base, openai_api_key=openai_api_key)
resp = chat_model([humanmessage(content="给我一个django admin的demo代码")])
chat_model.predict("你叫什么?")

# /v1/chat/completions普通响应
chat_model = chatopenai(openai_api_base=openai_api_base, openai_api_key=openai_api_key)
resp = chat_model.predict("给我一个django admin的demo代码")
print(resp)

# /v1/completions流式响应
llm = openai(streaming=true, callbacks=[streamingstdoutcallbackhandler()], temperature=0, openai_api_base=openai_api_base, openai_api_key=openai_api_key)
llm("登鹳雀楼->王之涣\n夜雨寄北->")

# /v1/completions普通响应
llm = openai(openai_api_base="http://43.134.77.153:5000/v1", openai_api_key=openai_api_key)
print(llm("登鹳雀楼->王之涣\n夜雨寄北->"))
```
  
openai translator 设置中把api url修改为你的ip  

<a href="https://s2.loli.net/2023/09/25/jbqns1kbljhv4k6.png" target="_blank"><img src="https://s2.loli.net/2023/09/25/jbqns1kbljhv4k6.png" width="60%"></a>  

