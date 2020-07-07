# Adversarial Neural Pruning with Latent Vulnerability Suppression
This is the implementation of [Adversarial Neural Pruning with Latent Vulnerability Suppression](https://arxiv.org/pdf/1908.04355.pdf).

**Authors**: [Divyam Madaan](https://dmadaan.com/), [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html), [Sung Ju Hwang](http://sungjuhwang.com/)

## Abstract
<img align="middle" width="700" src="https://github.com/divyam3897/ICML_ANP/blob/master/concept.png">

Despite the remarkable performance of deep neural networks on various computer vision tasks, they are known to be susceptible to adversarial perturbations, 
which makes it challenging to deploy them in real-world safety-critical applications. In this paper, we conjecture that the leading cause of adversarial vulnerability 
is the distortion in the latent feature space, and provide methods to suppress them effectively. Explicitly, we define vulnerability for each latent feature and then 
propose a new loss for adversarial learning, Vulnerability Suppression (VS) loss, that aims to minimize the feature-level vulnerability during training. 
We further propose a Bayesian framework to prune features with high vulnerability to reduce both vulnerability and loss on adversarial samples. 
We validate our Adversarial Neural Pruning with Vulnerability Suppression (ANP-VS) method on multiple benchmark datasets, on which it not only obtains state-of-the-art 
adversarial robustness but also improves the performance on clean examples, using only a fraction of the parameters used by the full network. 
Further qualitative analysis suggests that the improvements come from the suppression of feature-level vulnerability.

__Contribution of this work__
- We hypothesize that the distortion in the latent features as the leading cause of DNN's susceptibility to adversarial attacks and formally describe the concepts of the vulnerability of latent-features based on the expectation of the distortion of latent-features with respect to input perturbations.
- Based on this finding, we introduce a novel defense mechanism, *Adversarial Neural Pruning with Vulnerability Suppression (ANP-VS)*, to mitigate the feature-level vulnerability. The resulting framework learns a Bayesian pruning (dropout) mask to prune out the vulnerable features while preserving the 
robust ones by minimizing the adversarial and *{Vulnerability Suppression (VS)* loss. 
- We experimentally validate our proposed method on MNIST, CIFAR-10, and CIFAR-100 datasets, on which it achieves state-of-the-art robustness with a substantial reduction in memory and computation, with qualitative analysis which suggests that the improvement on robustness comes from its suppression of feature-level vulnerability


## Prerequisites
- Python 3.5
- Tensorflow 1.14.0
- CUDA 10.0
- cudnn 7.6.5

## Training
### MNIST dataset
  * Train base model
  ```
  $ python -m ANP_VS.src.experiments.run --net lenet_conv --mode base --data mnist
  ```

  * Train ANP-VS
  ```
  $ python -m ANP_VS.src.experiments.run --net lenet_conv --mode bbd --eps=0.3 --step_size=0.01 --adv_train=True --pgd_steps=20 --vulnerability=True --data mnist --n_epochs 200 --beta_weight=4 --lambda_weight=0.001
  ```

### CIFAR datasets
  * Train base model
  ```
  $ python -m ANP_VS.src.experiments.run --net vgg16 --mode base --data cifar10
  $ python -m ANP_VS.src.experiments.run --net vgg16 --mode base --data cifar100
  ```

  * Train ANP-VS
  ```
  # CIFAR-10
  $ python -m ANP_VS.src.experiments.run --net vgg16 --mode bbd --eps=0.03 --step_size=0.007 --adv_train=True --pgd_steps=10 --vulnerability=True --data cifar10 --n_epochs 200 --beta_weight=2 --lambda_weight=0.0001   
  
  # CIFAR-100 
  $ python -m ANP_VS.src.experiments.run --net vgg16 --mode bbd --eps=0.03 --step_size=0.007 --adv_train=True --pgd_steps=10 --vulnerability=True --data cifar100 --n_epochs 200 --beta_weight=1 --lambda_weight=0.1
  ```
  
## Evaluation

### Attack using PGD attack
```
# MNIST
$ python -m ANP_VS.src.experiments.run --net lenet_conv --mode bbd --eval_mode attack --white_box=True --attack_source base --eps 0.3 --step_size 0.01 --pgd_steps=40 --vulnerability_type=expected --data mnist --adv_train=True

# CIFAR-10
$ python -m ANP_VS.src.experiments.run --net vgg16 --mode bbd --eval_mode attack --white_box=True --attack_source base --eps 0.03 --step_size 0.007 --pgd_steps=40 --vulnerability_type=expected --data cifar10 --adv_train=True

# CIFAR-100
$ python -m ANP_VS.src.experiments.run --net vgg16 --mode bbd --eval_mode attack --white_box=True --attack_source base --eps 0.03 --step_size 0.007 --pgd_steps=40 --vulnerability_type=expected --data cifar100 --adv_train=True
```
For black-box attack, train a base adversarial training model and set `--white_box=False`

###  Computational efficiency
```
# MNIST
$ python -m ANP_VS.src.experiments.run --net lenet_conv --mode bbd --eval_mode test --data mnist --adv_train=True

# CIFAR-10
$ python -m ANP_VS.src.experiments.run --net vgg16 --mode bbd --eval_mode test --data cifar10 --adv_train=True

# CIFAR-100
$ python -m ANP_VS.src.experiments.run --net vgg16 --mode bbd --eval_mode test --data cifar100 --adv_train=True
```

## Results
The results in the main paper (average over five independent runs). The best results of adversarial baselines are highlighted in bold. 🡑(🡓) indicates that the higher (lower) number is the better.

### MNIST dataset on Lenet5-Caffe architecture

|  Model     | Clean Acc. (🡑) | White Box Adversarial acc. (🡑) | Black Box Adversarial acc. (🡑) | White Box Vulnerability (🡓) | Black Box Vulnerability (🡓) | Memory (🡓) | Flops (🡑) | Sparsity (🡑) |
| ------| ---------------- | ----------------- | ------------------ | ------------------- |------------------------|---------------|---------------|---------------|
| Standard |  99.29 | 0.00 | 8.02 | 0.129 | 0.113 | 100.0 |1.00 | 0.00
| BP |  99.34 | 0.00 | 12.99 | 0.091 |0.078 | 4.14 | 9.68 |83.48 
| AT |  99.14 | 88.03 | 94.18 | 0.045 | 0.040 | 100.0 |1.00 | 0.00
| AT BNN | 99.16 | 88.44 | 94.87 | 0.364 | 0.199 |  200.0 | 0.50 | 0.00
| Pretrained AT|  __99.18__ | 88.26 | 94.49 | 0.412 | 0.381 | 100.0 |1.00 | 0.00
| ADMM |  99.01| 88.47|94.61| 0.041| 0.038 |100.0 |1.00| 80.00
| TRADES |  99.07| 89.67| 95.04| 0.037| 0.033| 100.0| 1.00| 0.00
| ANP_VS | 99.05| __91.31__| __95.43__| __0.017__| __0.015__| __6.81__| __10.57__| __84.16__

### CIFAR-10 dataset on VGG-16 architecture


|  Model     | Clean Acc. (🡑) | White Box Adversarial acc. (🡑) | Black Box Adversarial acc. (🡑) | White Box Vulnerability (🡓) | Black Box Vulnerability (🡓) | Memory (🡓) | Flops (🡑) | Sparsity (🡑) |
| ------| ---------------- | ----------------- | ------------------ | ------------------- |------------------------|---------------|---------------|---------------|
| Standard |  92.76| 13.79| 41.65| 0.077| 0.065| 100.0| 1.00| 0.00
| BP |  92.91| 14.30| 42.88| 0.037| 0.033| 12.41| 2.34| 75.92
| AT |  87.50| 49.85| 63.70| 0.050| 0.047| 100.0| 1.00| 0.00
| AT BNN | 86.69| 51.87| 64.92|0.267| 0.238| 200.0| 0.50| 0.00
| Pretrained AT|  87.50| 52.25| 66.10| 0.041| 0.036| 100.0| 1.00| 0.00
| ADMM |  78.15 |47.37 |62.15 |0.034| 0.030| 100.0| 1.00| 75.00
| TRADES |  80.33| 52.08| 64.80| 0.045| 0.042| 100.0| 1.00| 0.00
| ANP-VS | __88.18__| __56.21__| __71.44__| __0.019__| __0.016__| __12.27__| __2.41__| __76.53__


### CIFAR-100 dataset on VGG-16 architecture
|  Model     | Clean Acc. (🡑) | White Box Adversarial acc. (🡑) | Black Box Adversarial acc. (🡑) | White Box Vulnerability (🡓) | Black Box Vulnerability (🡓) | Memory (🡓) | Flops (🡑) | Sparsity (🡑) |
| ------| ---------------- | ----------------- | ------------------ | ------------------- |------------------------|---------------|---------------|---------------|
| Standard |   67.44  |2.81 | 14.94 | 0.143  |0.119 | 100.0| 1.00| 0.00
| BP |   69.40 | 3.12| 16.39 | 0.067 | 0.059 | 18.59 | 1.95 | 63.48
| AT |   57.79 | 19.07 | 32.47 | 0.079| 0.071| 100.0| 1.00| 0.00
| AT BNN |  53.75 | 19.40 | 30.38 | 0.446 | 0.385 | 200.0| 0.50| 0.00
| Pretrained AT|   57.14  |19.86 | 35.42 | 0.071 | 0.065 | 100.0| 1.00| 0.00
| ADMM |   52.52 |19.65 | 31.30 | 0.060 | 0.056 | 100.0| 1.00| 75.00
| TRADES |   56.70 | 21.21 | 32.81 | 0.065 |0.060 | 100.0| 1.00| 0.00
| ANP-VS |  __59.15__  |__22.35__ | __37.01__ | __0.035__ | __0.030__ | __16.74__ | __2.02__ | __66.80__


## Contributing
We'd love to accept your contributions to this project. Please feel free to open an issue, or submit a pull request as necessary. If you have implementations of this repository in other ML frameworks, please reach out so we may highlight them here.

## Acknowledgment
The code is build upon [OpenXAIProject/Variational_Dropouts](https://github.com/OpenXAIProject/Variational_Dropouts)


## Citation
If you found the provided code useful, please cite our work.

```
@inproceedings{
    madaan2020adversarial,
    title={Adversarial Neural Pruning with Latent Vulnerability Suppression},
    author={Divyam Madaan and Jinwoo Shin and Sung Ju Hwang},
    booktitle={ICML},
    year={2020}
}
```
