## Towards Unbiased Training in Federated Open-world Semi-supervised Learning

This repo contains the reference source code in PyTorch of the FedoSSL algorithm. For more details please check our paper [Towards Unbiased Training in Federated Open-world Semi-supervised Learning](https://arxiv.org/abs/2305.00771) (ICML '23). 

### Dependencies

The code is built with the following libraries:

- [PyTorch==1.9](https://pytorch.org/)
- [sklearn==1.0.1](https://scikit-learn.org/)

### Usage

##### Get Started

We use SimCLR for pretraining. The weights used in our paper can be downloaded in this [link](https://drive.google.com/file/d/19tvqJYjqyo9rktr3ULTp_E33IqqPew0D/view?usp=sharing).

- To train on CIFAR-100, run

```bash
python fedossl_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5
```

- To train on ImageNet-100, first use ```gen_imagenet_list.py``` to generate corresponding splitting lists, then run

```bash
python fedossl_imagenet.py --labeled-num 50 --labeled-ratio 0.5
```
