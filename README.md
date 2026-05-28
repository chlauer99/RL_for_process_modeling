# RL_for_process_modeling

This repository contains the implementation and resources for the paper:

**"Improving LLM-Generated Process Model Quality Through Reinforcement Learning: The Role of Reward Function Design"**

## Reinforcement Learning

The folder `reinforcement_learning` contains the scripts used for training Large Language Models (LLMs) with Reinforcement Learning (RL).

To compute the reward functions, the following external repository must additionally be cloned and included in the project:

* BEF4LLM: https://gitlab-iwi.dfki.de/lauer/bef4llm

The required implementation can be found in:

```python
bef4llm/src/bef4llm/reward_functions.py
```

## Dataset

The dataset used for training is publicly available on Hugging Face:

* https://huggingface.co/datasets/chlauer/Signavio_text_bpmn

The dataset consists of a single JSON file containing all training instances. Each entry includes the following fields:

* `instruction`
* `input`
* `output`
* `system`
* `language`

The dataset contains textual process descriptions paired with corresponding BPMN process models.
