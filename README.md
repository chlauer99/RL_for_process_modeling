# RL for process modeling

This repository contains the implementation and resources for the paper:

**Improving LLM-Generated Process Model Quality Through Reinforcement Learning: The Role of Reward Function Design**

## Supervised Finetuning

The folder `supervised_finetuning` contains the scripts used for Supervised Finetuning (SFT) using LLamaFactory.

## Reinforcement Learning

The folder `reinforcement_learning` contains the notebook used for Reinforcement Learning (RL) using unsloth.

## Statistical evaluation

The folder `statistical evaluation` contains the source code used for the statistical evaluation.

## External dependencies
The following external repositories must additionally be cloned/installed:

* [BEF4LLM](https://gitlab-iwi.dfki.de/lauer/bef4llm)

The required implementation can be found in *bef4llm/src/bef4llm/reward_functions.py*

* [LLamaFactory](https://github.com/hiyouga/LlamaFactory)

For guidance on how to perform SFT, see [docs](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html)

* [unsloth](https://github.com/unslothai/unsloth/)

For guidance on how to perform GSPO, see [docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/gspo-reinforcement-learning)

 
## Dataset

The dataset used for training is publicly available on [Hugging Face](https://huggingface.co/datasets/chlauer/Signavio_text_bpmn).

The dataset consists of a single JSON file containing all training instances. Each entry includes the following fields:

* `instruction`
* `input`
* `output`
* `system`
* `language`

The dataset contains textual process descriptions paired with corresponding BPMN process models.