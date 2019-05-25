# Transfer of Samples in Policy Search via Multiple Importance Sampling

This repository contains the code for our paper "Transfer of Samples in Policy Search via Multiple Importance Sampling", which will appear in ICML 2019. The full version of the paper is available [here](https://home.deib.polimi.it/atirinzoni/files/tirinzoni2019transfer.pdf).

### Abstract

We consider the transfer of experience samples in reinforcement learning. Most of the previous works in this context focused on value-based settings, where transferring instances conveniently reduces to the transfer of (s, a, s', r) tuples. In this paper, we consider the more complex case of reusing samples in policy search methods, in which the agent is required to transfer entire trajectories between environments with different transition models. By leveraging ideas from multiple importance sampling, we propose robust gradient estimators that effectively achieve this goal, along with several techniques to reduce their variance. In the case where the transition models are known, we theoretically establish the robustness to the negative transfer for our estimators. In the case of unknown models, we propose a method to efficiently estimate them when the target task belongs to a finite set of possible tasks and when it belongs to some reproducing kernel Hilbert space. We provide empirical results to show the effectiveness of our estimators.

### Repository Structure

The repository is still **under preparation**. We are cleaning, reformatting, and commenting the code. Instructions on how to use it will be published soon.
