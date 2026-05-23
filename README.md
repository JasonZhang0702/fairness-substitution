# fairness-interpretability

[add]

[7-opensourced-llm.json](result%2F7-opensourced-llm.json): 7个开源模型的实验结果
```
model_name: {
    method: {
        abscore_avg: xxx,
        abscore_dist: [xxx, xxx, xxx, xxx],
        diffaware_avg: xxx,
        diffaware_dist: [xxx, xxx, xxx, xxx],
        ctxtaware_avg: xxx,
        ctxtaware_dist: [xxx, xxx, xxx, xxx],
    }
}
[Tips]: base 是直接回答；debiasing/none/unbias-re 是自我反思基线
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
