# ChatML
在 ChatML 中，对话被分为以下几层或角色：

* 系统 system
* 助手 assistant
* 用户 user
  
这只是 ChatML 的 0 版本，我们承诺将对该语言进行重大开发。

ChatML 中容纳的有效载荷目前仅为文本。OpenAI 预计将引入其他数据类型。这符合大型基础模型的概念，即将开始结合文本、图像、声音等。

用户仍然可以使用不安全的原始字符串格式。但这种格式本身就允许注入。

OpenAI 处于以负责任的方式引导和管理 LLM 领域的理想位置。为创建应用程序制定基础标准。

**ChatML 向模型明确了每段文本的来源，并特别显示了人类文本和人工智能文本之间的界限**。

这提供了缓解并最终解决注入问题的机会，因为模型可以分辨出哪些指令来自开发人员、用户或其自己的输入。〜OpenAI

## ChatML 示例代码

system下面是 ChatML 示例 JSON 文件，其中定义了、user和的角色assistant。

``` json
[ { "role" :  "system" ,  
      "content"  :  "您是 ChatGPT，这是 OpenAI 训练的大型语言模型。请尽可能简洁地回答。\n知识截止时间：2021-09-01\n当前日期：2023-03-02" } , 
 { "role" :  "user" ,  
      "content"  :  "您好吗？" } , 
 { "role" :  "assistant" , "  
      content"  :  "我很好" } , 
 { "role" :  "user" ,  
      "content"  :  "OpenAI 公司的使命是什么？" } ]
```

有效的 Python 代码片段如下：
``` shell 
pip install openai 
```

``` python
import os 
import openai 
openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

 finish = openai.ChatCompletion.create( 
  model= "gpt-3.5-turbo" , 
  messages = [{ "role" : "system" , "content" : "您是 ChatGPT，这是由 OpenAI 训练的大型语言模型。请尽可能简洁地回答。\n知识截止：2021-09-01\n当前日期：2023-03-02" }, 
{ "role" : "user" , "content" : "您好吗？" }, 
{ "role" : "assistant" , "content" : "我很好" }, 
{ "role" : "user" , "content" : "OpenAI 公司的使命是什么？" }] 
) 
#print(completion) 
print (completion)
```
通过下面的输出，注意定义的角色、模型细节gpt-3.5-turbo-0301和其他细节。

``` json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The mission of OpenAI is to ensure that artificial intelligence (AI) benefits humanity as a whole, by developing and promoting friendly AI for everyone, researching and mitigating risks associated with AI, and helping shape the policy and discourse around AI.",
        "role": "assistant"
      }
    }
  ],
  "created": 1677751157,
  "id": "chatcmpl-6pa0TlU1OFiTKpSrTRBbiGYFIl0x3",
  "model": "gpt-3.5-turbo-0301",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 50,
    "prompt_tokens": 84,
    "total_tokens": 134
  }
}
```

## 结束语
构建基于 LLM 的对话界面的挑战之一是将提示节点排序成链的概念。

由于输入的非结构化性质，节点之间的边很难管理。输入通常是自然语言或对话，本质上是非结构化的。

ChatML 将极大地帮助创建提交到链的数据转换的标准目标。

## 参考
1. https://cobusgreyling.medium.com/the-introduction-of-chat-markup-language-chatml-is-important-for-a-number-of-reasons-5061f6fe2a85
2. https://zhuanlan.zhihu.com/p/666461139