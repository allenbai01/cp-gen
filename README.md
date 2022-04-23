# Conformal Prediction with General Function Classes

This repository provides the implementation for the paper [Efficient and Differentiable Conformal Prediction with General Function Classes](https://arxiv.org/abs/2202.11091).

## Overview
Our algorithm `CP-Gen` (Conformal Prediction with General Function Classes) is a generalization of conformal prediction to learning multiple parameters. `CP-Gen` can learn within an arbitrary family of prediction sets, by solving the constrained ERM problem of best efficiency subject to valid empirical coverage. Our code implements the recalibrated version `CP-Gen-Recal` to achieve valid finite-sample coverage.


<p align="center">
  <img src="https://github.com/allenbai01/cp-gen/blob/main/figures/fig1_left.png" width="40%" hspace="5%">
  <img src="https://github.com/allenbai01/cp-gen/blob/main/figures/fig1_right.png" width="40%">
  <figcaption>
    Illustration of our <tt>CP-Gen</tt> algorithm. While vanilla conformal prediction only learns a single parameter (within its conformalization step) by a simple thresholding rule over a coverage-efficiency curve (Left), <tt>CP-Gen</tt> is able to further improve the efficiency by thresholding a region formed by a larger function class (Right).
</figcaption>
</p>


## Install requirements

```
pip install -r requirements.txt
```

## Minimum-volume prediction set fo multi-output regression

```
chmod +x ./multi_output_run.sh
./multi_output_run.sh
```

Note: To download the dataset for this task, use `git lfs clone` to clone this repository. See https://git-lfs.github.com/ for the installation guide of `git lfs`.

## Improved prediction intervals via conformal quantile finetuning

```
chmod +x ./conformal_finetuning_run.sh
./conformal_finetuning_run.sh
```

## Label prediction sets on ImageNet

```
cd imagenet
chmod +x ./imagenet_run.sh
./imagenet_run.sh
```

## Miscellanous
Part of the code is built upon the following codebases:

[cqr](https://github.com/yromano/cqr)\
[conformal_classification](https://github.com/aangelopoulos/conformal_classification)\
[Mujoco](https://github.com/deepmind/mujoco)

If you use this code in your research, please cite our paper
```
@inproceedings{bai2022efficient,
  title={Efficient and Differentiable Conformal Prediction with General Function Classes},
  author={Yu Bai and Song Mei and Huan Wang and Yingbo Zhou and Caiming Xiong},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=Ht85_jyihxp}
}
```

