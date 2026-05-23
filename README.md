# fairness-interpretability

[add]

[7-opensourced-llm.json](result%2F7-opensourced-llm.json): 7个开源模型的实验结果
```
model_name: {
    method: {
        first_principle: [xxx, xxx, xxx, xxx] // 原则判断的准确率（二分类）
        abscore_avg: xxx, // 绝对分数平均值
        abscore_dist: [xxx, xxx, xxx, xxx], // 绝对分数分布
        diffaware_avg: xxx,
        diffaware_dist: [xxx, xxx, xxx, xxx],
        ctxtaware_avg: xxx,
        ctxtaware_dist: [xxx, xxx, xxx, xxx],
    }
}
[Tips]: 
1. base 是直接回答；debiasing/none/unbias-re 是自我反思基线；rawls 是对两阶段的消融；
2. rawls-cott(-nodefine) 是 distinguish-judge 方法，后面跟 -gt 表示上限，-reverse 表示下限
3. 这里正文汇报 rawls-cott-nodefine，至于 rawls-cott 作为验证消融
4. 指标分析需要四个 benchmark 分开讨论，优先考虑 diffaware 再是 ctxtaware。具体可参考：https://aclanthology.org/2025.acl-long.341/

```
[fairness-neuron](result%2Ffairness-neuron): 7个开源模型的公平神经元
```
{
only_difference_neurons: {lid_nid}
only_liberty_neurons:
base_neurons: 
}
lid: 网络层编号，nid: 该网络层下的神经元编号
```

1. 推理脚本
```shell
python inference-test-fairdiff.py
```
参数解释：
- prompt: 不同的提示指令，结构化保存在 `PROMPT` 变量中。例如：`rawls-cott-nodefine` 表示依赖模型自身，先区分再决策，`rawls-cott-nodefine-gt` 表示修改模型区分输出为 ground truth，再决策，`rawls-cott-nodefine` 表示修改模型区分输出为与 ground truth 相反的结果，再决策
- think: 针对 Qwen3 等推理模型。默认为 False，否则输出太长了。

2. 评估脚本
```shell
python evaluate.py --model_name Qwen2.5-14B-Instruct --prompt rawls-cott-nodefine
```
