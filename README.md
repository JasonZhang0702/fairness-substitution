# fairness-interpretability


1. 推理脚本
```shell
python inference-test-fairdiff.py
```
参数解释：
- prompt: 不同的提示指令。结构化保存在 `PROMPT` 变量中。
- think: 针对 Qwen3 等推理模型。默认为 False，否则输出太长了。

2. 评估脚本
```shell
python evaluate.py
```
