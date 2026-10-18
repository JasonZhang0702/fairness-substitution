### Do LLMs Really Lack Difference Awareness? Uncovering *Heuristic Substitution* in Group-Differentiating Fairness

> In this paper, we investigate the issue of **difference awareness** in LLMs, defined as a model’s ability to recognize and acknowledge meaningful differences between social groups in fairness decision-making. Previous studies have consistently shown that models perform poorly on such tasks, a finding commonly attributed to a lack of difference awareness. 
> We propose an alternative account, which we term **Fairness Heuristic Substitution**: difference awareness is not necessarily absent, but the complex demand of simultaneously identifying relevant group differences and determining their implications for fairness decisions may lead models to substitute the simpler, more accessible equal-treatment heuristic, i.e., **formal equality**. 
> Based on this hypothesis, we introduce **TDD**, a **T**wo-stage **D**istinguish-**D**ecide framework that explicitly decomposes fairness reasoning into first distinguishing contextually relevant group differences and then making the corresponding decision, thereby discouraging reliance on formal-equality shortcuts.
> 1. Across four benchmarks and ten models ranging from billions to trillions of parameters, TDD significantly outperforms one-stage baselines, even one strengthened through simulated test-set leakage.
> 2. Combining TDD with neuron-level contrastive probing provides mechanistic evidence consistent with the **Fairness Heuristic Substitution** hypothesis by revealing two signatures of formal equality: a compact, locally coordinated representational structure and a pronounced activation bias under default reasoning.
> 3. TDD’s stage-wise decomposition uncovers overlooked failure modes in fairness reasoning, including heuristic displacement and misaligned stage dynamics.


Run the following script to reproduce the result visualizations presented in the paper.
```python
python diffaware_mainresult_visualization.py
```