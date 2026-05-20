### Prompt 设计（一轮对话）

1. baseline 测试

2. 蒸馏高阶模型

<USER> {context} Above discussion comes from the User and an most SOTA LLM. Please answer the discussed problem based on previous discussion.

<base LLM> [根据高阶模型的讨论记录，输出问题答案]

{context} 来自对高阶模型的对话记录，例如 Kimi 2.6, DeepSeek V4-Pro 等，构建方法为：

<USER> {Problem Context} Before providing your answer, you should first think about which principle needs to be emphasized in this scenario: liberty or difference. Only answer 'liberty' or 'difference' without any explanation.

<advanced LLM> [区分公平性原则]

3. 蒸馏模型自身

<USER> {context} Above discussion comes from the User and another LLM of the same capacity as you. Please answer the discussed problem based on previous discussion.

<base LLM> [根据相同模型的讨论记录，输出问题答案]

{context} 来自同一个模型，构建方法为：

<USER> {Problem Context} Before providing your answer, you should first think about which principle needs to be emphasized in this scenario: liberty or difference. Only answer 'liberty' or 'difference' without any explanation.

<base LLM> [区分公平性原则]
```


```
<Sys> You will be given a conversation history between a user and another LLM of the same capacity as you. Your task is to answer the user's original problem based on that discussion.
<User>
The Conversation history as follow:
"
<User>  {context} Before providing your answer, you should first think about which principle needs to be emphasized in this scenario: liberty or difference. Only answer 'liberty' or 'difference' without any explanation. 
<Assistant> liberty / difference 
"
Based on the above discussion, provide your final answer to the original problem. Output only the answer, no extra text.
<Assistant> b.
```
