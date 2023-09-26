# my_openai_api

部署你自己的**OpenAI** 格式api😆，基于**flask, transformers** (使用 **Baichuan2-13B-Chat-4bits** 模型，可以运行在单张Tesla T4显卡) ，实现以下**OpenAI**接口：
- **Chat**   /v1/chat/completions
- **Models**   /v1/models
- **Completions**   /v1/completions

同时实现接口相应的STREAMING模式，保证在**langchain**中基础调用

## 起因

目前Baichuan2-13B-Chat int4量化后可在单张tesla T4显卡运行，并且效果和速度还可以，可以和gpt-3.5媲美。  
- **Baichuan2-13B-Chat-4bits**：[https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat-4bits](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat-4bits)

## 最低配置

需要16g显存，如果主机显存不够可以考虑腾讯云的活动，60块钱15天32g内存、T4显卡的主机，非常划算😝，可以跑动baichuan2-13b-chat-4bits。  
  
<a href="https://s2.loli.net/2023/09/25/q7C4jdJocwym1fh.png" target="_blank"><img src="https://s2.loli.net/2023/09/25/q7C4jdJocwym1fh.png" width="60%"></a>  

地址: [https://cloud.tencent.com/act/pro/gpu-study](https://cloud.tencent.com/act/pro/gpu-study)  

如果想要本地运行，T4显卡价格在5600元左右，也可以考虑2080ti魔改22g版本，某宝只要2600元左右 🤓️。

## 安装

1. 下载代码
```
git clone https://github.com/billvsme/my_openai_api.git
```
2. 下载Baichuan2-13B-Chat-4bits模型
```
cd my_openai_api

git lfs install #需要先安装好git-lfs
git clone https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat-4bits
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

![github_my_open_api_002.png](https://s2.loli.net/2023/09/25/o8I5GE3ONfhSaqz.png)


## 使用

### langchain

替换openai_base_api
```
# coding: utf-8
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import (
    HumanMessage,
)

openai_api_base = "http://127.0.0.1:5000/v1"
openai_api_key = "test"

# /v1/chat/completions流式响应
chat_model = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], openai_api_base=openai_api_base, openai_api_key=openai_api_key)
resp = chat_model([HumanMessage(content="给我一个django admin的demo代码")])
chat_model.predict("你叫什么?")

# /v1/chat/completions普通响应
chat_model = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key=openai_api_key)
resp = chat_model.predict("给我一个django admin的demo代码")
print(resp)

# /v1/completions流式响应
llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, openai_api_base=openai_api_base, openai_api_key=openai_api_key)
llm("登鹳雀楼->王之涣\n夜雨寄北->")

# /v1/completions普通响应
llm = OpenAI(openai_api_base=openai_api_base, openai_api_key=openai_api_key)
print(llm("登鹳雀楼->王之涣\n夜雨寄北->"))
```

### ChatGPT Next 

设置中把接口地址修改为你的ip，如果部署网页为https，注意在Chrome设置中“不安全内容”选择“允许”

<a href="https://sm.ms/image/8eMUw6sHXP9QBmj" target="_blank"><img src="https://s2.loli.net/2023/09/25/8eMUw6sHXP9QBmj.png" width="50%"></a>

### OpenAI Translator

设置中把api url修改为你的ip  

<a href="https://s2.loli.net/2023/09/25/q7C4jdJocwym1fh.png" target="_blank"><img src="https://s2.loli.net/2023/09/25/jbqNs1kBlJHv4K6.png" width="60%"></a>  

