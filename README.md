# CodeAttack: Code-based Adversarial Attacks for Pre-Trained Programming Language Models

This repository contains the code for the AAAI 2023 paper [CodeAttack: Code-based Adversarial Attacks for Pre-Trained Programming Language Models](https://arxiv.org/pdf/2206.00052.pdf).

## Overview

Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models operate in the natural channel of code, i.e., they are primarily concerned with the human understanding of the code. They are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks in the natural channel. We propose, CodeAttack, a simple yet effective blackbox attack model that uses code structure to generate effective, efficient, and imperceptible adversarial code samples and demonstrates the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall drop in performance while being more efficient, imperceptible, consistent, and fluent. 


## Run

To install the dependencies please execute the command ```pip install -r requirements.txt```. To run the code, please execute ```python codeattack.py [args]``` with the follwing arguments:

|Argument |Description|
|--- |--- |
|--attack_model | Model that attacks |
|--victim_model | Model to attack |
|--task | The task to attack |
|--lang | The input programming language dataset [java_cs, cs_java, java_small] |
|--use_ast | A boolean flag for whether to use AST constraint |
|--use_dfg | A boolean flag for whether or not to use DFG as a constraint|
|--out_dirname | The output directory |
|--input_lang | The input programming language (only required for GraphCodeBERT)|
|--use_imp | A boolean flag to either attack random words or attack only important/vulnerable words|
|--theta | The percentage of tokens to attack|




Code for this repository has been adapted from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE), [CodeBERT)[https://github.com/microsoft/CodeBERT], and [TextFooler](https://github.com/jind11/TextFooler).

## Citation

```
@article{jha2022codeattack,
  title={Codeattack: Code-based adversarial attacks for pre-trained programming language models},
  author={Jha, Akshita and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2206.00052},
  year={2022}
  }
```
