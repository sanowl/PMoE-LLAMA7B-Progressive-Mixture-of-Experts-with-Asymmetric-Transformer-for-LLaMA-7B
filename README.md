# PMoE-LLAMA7B: Progressive Mixture of Experts with Asymmetric Transformer for llama/Meta-Llama-3.1-8B

This repository contains the implementation of PMoE (Progressive Mixture of Experts with Asymmetric Transformer) for the llama/Meta-Llama-3.1-8B, as described in the paper "PMoE: Progressive Mixture of Experts with Asymmetric Transformer for Continual Learning" by Min Jae Jung and JooHee Kim.

## Introduction

Large Language Models (LLMs) encounter significant challenges in continual learning due to catastrophic forgetting, where new information overwrites previously acquired knowledge. This limitation leads to substantial environmental and economic waste. PMoE aims to minimize forgetting by utilizing an asymmetric design with shallow layers dedicated to general knowledge and deep layers for new knowledge. PMoE incorporates progressively added experts in deep layers and a router that allocates new knowledge to the appropriate experts efficiently. Extensive experiments on TRACE datasets and general language understanding datasets demonstrate that the proposed PMoE outperforms previous state-of-the-art approaches.

## Key Features

- **Asymmetric Depth Design**: Shallow layers retain general knowledge, while deeper layers acquire new task-specific knowledge.
- **Progressive Expert Addition**: Experts are progressively added in the deep layers to handle new tasks.
- **Efficient Routing**: The router efficiently allocates new knowledge to the appropriate experts using deep features.

## Model Architecture

The PMoE model is built on top of the llama/Meta-Llama-3.1-8B and integrates Low-Rank Adaptation (LoRA) as the expert component. The architecture consists of:

1. **Shallow Layers**: General knowledge retention using frozen layers.
2. **Deep Layers**: Task-specific knowledge acquisition using LoRA-based experts.
3. **Router**: Allocates input text to the appropriate experts based on deep features.
