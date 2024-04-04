This is code for the submission "A New Memory-based Continual Learning Algorithm with Provable Learning and Forgetting Guarantees".

This version of code is based on [DER](https://github.com/aimagelab/mammoth) and we extened to our algorithm STREAM.

## Abstract

Continual Learning (CL) is a learning paradigm that aims to train a model across a sequence of tasks, where the model retains old knowledge while adapting to new environments. To avoid catastrophic forgetting, memory-based CL methods were proposed, which store relevant information from old tasks and use this information to guide the learning of new tasks. These methods update the model by incorporating information from both the current gradient and the memory gradient in every iteration. In this paper, we show that existing memory-based CL methods do not have learning and forgetting guarantees on a simple counterexample. To address this issue, we designed a new algorithm called STREAM, which directly solves the constrained optimization problem of minimizing the loss of current tasks under the constraint of small forgetting in previous tasks. Unlike existing approaches that use both the current gradient and the memory gradient to update the model, STREAM updates the model using either the current gradient or the memory gradient. The branch choice relies on a moving-average estimator for forgetting. Under standard assumptions of the objective function, we prove that the STREAM algorithm can converge to a nearly $\epsilon$-KKT point of the constrained optimization problem within the polynomial number of stochastic gradient evaluations, which provides provable learning and forgetting guarantees. We conducted extensive experiments in various CL settings, including single- and multi-modal classification tasks, to show that our proposed algorithm significantly outperforms strong baselines in CL. 

##Requirements
PyTorch >= v1.6.0. 

## Setup
- Use function `main.py` to run all the experiments.

- To reproduce the results, use the best argument `--load_best_args` to load the best hyperparameters for each compared algorithms from the paper.

## Sequential Tiny-imagenet CL experiments 
we set buffer size to 200 for all the memory-based methods for fair comparison.
You could download Tiny-Imagenet data from [link](https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view) and put it in datasets directory

Please replace `model` with any one of `agem, mas ogd, der, ewc_on, gdumb, mega, stream ` for running the code

`python main.py --dataset seq-tinyimg --model model --buffer_size 200 --load_best_args`
