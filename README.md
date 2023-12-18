<div align="center">
  <a href="http://zhijian.readthedocs.io"><img width="450px" height="auto" src="assests/Model-Spider.png"></a>
</div>


&nbsp;

<div align="center">
    <img src="https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge" alt="Generic badge", height="21">
    <img src="https://img.shields.io/github/actions/workflow/status/zhangyikaii/Model-Spider/tests.yml?branch=main&style=for-the-badge" alt="GitHub Workflow Status (branch)", height="21">
    <br>
    <img src="https://img.shields.io/pypi/v/ModelSpider?color=blue&style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI", height="21">
    <img src="https://img.shields.io/pypi/dm/ModelSpider?style=for-the-badge&color=blue" alt="PyPI - Downloads", height="21">
    <br>
    <img src="https://img.shields.io/badge/PYTORCH-1.4+-red?style=for-the-badge&logo=pytorch" alt="PyTorch - Version", height="21">
    <img src="https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white" alt="Python - Version", height="21">
</div>
<h3 align="center">
    <p>
        Model Spider: Learning to Rank Pre-Trained Models Efficiently (NeurIPS 2023 Spotlight)
    <p>
</h3>
<h4 align="center">
    <p>
        📑 <a href="https://arxiv.org/abs/2306.03900">[Paper]</a> [<b>Code</b>] <a href="TBD">[Blog]</a>
    <p>
    <p>
        <b>English</b> |
        <a href="https://github.com/zhangyikaii/Model-Spider/edit/main/README_CN.md">中文</a>
    <p>
</h4>

Figuring out which Pre-Trained Model (PTM) from a model zoo fits the target task is essential to take advantage of plentiful model resources. With the availability of **numerous heterogeneous PTMs from diverse fields**, efficiently selecting **the most suitable** PTM is challenging due to the time-consuming costs of carrying out forward or backward passes over all PTMs. In this paper, we propose **Model Spider**, which tokenizes both PTMs and tasks by **summarizing their characteristics into vectors to enable efficient PTM selection**.

By leveraging the **approximated performance of PTMs** on a separate set of training tasks, **Model Spider** learns to construct representation and measure the fitness score between **a model-task pair** via their representation. The ability to rank relevant PTMs higher than others generalizes to new tasks. With the top-ranked PTM candidates, we further learn to enrich task repr. with their PTM-specific semantics **to re-rank the PTMs for better selection**. **Model Spider** balances efficiency and selection ability, making PTM selection like a spider preying on a web.

**Model Spider** demonstrates promising performance across various model categories, including **visual models and Large Language Models (LLMs)**. In this repository, we have built a comprehensive and user-friendly PyTorch-based model ranking toolbox for evaluating the future generalization performance of models. It aids in selecting **the most suitable** foundation pre-trained models for achieving optimal performance in real-world tasks **after fine-tuning**. In this benchmark for selecting/ranking PTMs, we have replicated relevant model selection methods such as H-Score, LEEP, LogME, NCE, NLEEP, OTCE, PACTran, GBC, and LFC:

+ We introduce a *single-source model zoo*, building **10 PTMs** on ImageNet across five architecture families, *i.e.*, Inception, ResNet, DenseNet, MobileNet, and MNASNet. These models can be evaluated on **9 downstream datasets** using measure like *weighted tau*, including Aircraft, Caltech101, Cars, CIFAR10, CIFAR100, DTD, Pet, and SUN397 for classification, UTKFace and dSprites for regression.

+ We construct a *multi-source model zoo* where **42 heterogeneous PTMs** are pre-trained from multiple datasets, with 3 architectures of similar magnitude, *i.e.*, Inception-V3, ResNet-50, and DenseNet-201, pre-trained on 14 datasets, including animals, general and 3D objects, plants, scene-based, remote sensing, and multi-domain recognition. We evaluate the ability to select PTMs on Aircraft, DTD, and Pet datasets.

In this repo, you can figure out:

* Comprehensive implementations of existing model selection/ranking algorithms, along with 2 accompanying benchmark evaluations.
* Get started quickly with Model Spider, and enjoy its user-friendly inference capabilities.
* Feel free to customize the application scenarios of Model Spider, starting from pre-trained model libraries containing 10 and 42 models, respectively.

Also, if you meet problems, feel free to shoot us issues (English or Chinese)!

&nbsp;

## Table of Contents
- [Performance Evaluation](#pre-trained-model-ranking-performance)
- [Code Implementation](#code-implementation)
   - [Prerequisites](#prerequisites)
   - [Quick Start & Reproduce](#quick-start-&-reproduce)
   - [Train Your Own Model Spider](#train-your-own-model-spider)
- [Reproduce for Other Baselines](#reproduce-for-other-baselines)
- [Implementation for 42 models of Figure 3 in paper](#implementation-for-42-models-of-figure-3-in-paper)

&nbsp;

## Pre-trained Model Ranking Performance

Performance comparisons of **9 baseline approaches** and Model Spider on the *single-source model zoo* with weighted Kendall's tau. We denote the best-performing results in **bold**.

<table>
    <tr>
        <td><b>Method</b></td>
        <td colspan="10" align="center"><b>Downstream Target Dataset</b></td>
    </tr>
    <tr>
        <td><b>Weighted Tau</b></td>
        <td align="center">Aircraft</td>
        <td align="center">Caltech101</td>
        <td align="center">Cars</td>
        <td align="center">CIFAR10</td>
        <td align="center">CIFAR100</td>
        <td align="center">DTD</td>
        <td align="center">Pets</td>
        <td align="center">SUN397</td>
        <td align="center">Mean</td>
    </tr>
    <tr>
        <!-- <td><a href="https://arxiv.org/abs/1601.08188">H-Score</a></td> -->
        <td>H-Score </td> -->
        <td align="center">0.328</td>
        <td align="center">0.738</td>
        <td align="center">0.616</td>
        <td align="center">0.797</td>
        <td align="center">0.784</td>
        <td align="center">0.395</td>
        <td align="center">0.610</td>
        <td align="center">0.918</td>
        <td align="center">0.648</td>
    </tr>
    <tr>
        <td>NCE</td>
        <td align="center">0.501</td>
        <td align="center">0.752</td>
        <td align="center">0.771</td>
        <td align="center">0.694</td>
        <td align="center">0.617</td>
        <td align="center">0.403</td>
        <td align="center">0.696</td>
        <td align="center">0.892</td>
        <td align="center">0.666</td>
    </tr>
    <tr>
        <td>LEEP</td>
        <td align="center">0.244</td>
        <td align="center">0.014</td>
        <td align="center">0.704</td>
        <td align="center">0.601</td>
        <td align="center">0.620</td>
        <td align="center">-0.111</td>
        <td align="center">0.680</td>
        <td align="center">0.509</td>
        <td align="center">0.408</td>
    </tr>
    <tr>
        <td><span style="font-family:sans-serif;">N-</span>LEEP</td>
        <td align="center">-0.725</td>
        <td align="center">0.599</td>
        <td align="center">0.622</td>
        <td align="center">0.768</td>
        <td align="center">0.776</td>
        <td align="center">0.074</td>
        <td align="center">0.787</td>
        <td align="center">0.730</td>
        <td align="center">0.454</td>
    </tr>
    <tr>
        <td>LogME</td>
        <td align="center"><b>0.540</b></td>
        <td align="center">0.666</td>
        <td align="center">0.677</td>
        <td align="center">0.802</td>
        <td align="center">0.798</td>
        <td align="center">0.429</td>
        <td align="center">0.628</td>
        <td align="center">0.870</td>
        <td align="center">0.676</td>
    </tr>
    <tr>
        <td>PACTran</td>
        <td align="center">0.031</td>
        <td align="center">0.200</td>
        <td align="center">0.665</td>
        <td align="center">0.717</td>
        <td align="center">0.620</td>
        <td align="center">-0.236</td>
        <td align="center">0.616</td>
        <td align="center">0.565</td>
        <td align="center">0.397</td>
    </tr>
        <tr>
        <td>OTCE</td>
        <td align="center">-0.241</td>
        <td align="center">-0.011</td>
        <td align="center">-0.157</td>
        <td align="center">0.569</td>
        <td align="center">0.573</td>
        <td align="center">-0.165</td>
        <td align="center">0.402</td>
        <td align="center">0.218</td>
        <td align="center">0.149</td>
    </tr>
    <tr>
        <td>LFC</td>
        <td align="center">0.279</td>
        <td align="center">-0.165</td>
        <td align="center">0.243</td>
        <td align="center">0.346</td>
        <td align="center">0.418</td>
        <td align="center">-0.722</td>
        <td align="center">0.215</td>
        <td align="center">-0.344</td>
        <td align="center">0.034</td>
    </tr>
    <tr>
        <td>GBC</td>
        <td align="center">-0.744</td>
        <td align="center">-0.055</td>
        <td align="center">-0.265</td>
        <td align="center">0.758</td>
        <td align="center">0.544</td>
        <td align="center">-0.102</td>
        <td align="center">0.163</td>
        <td align="center">0.457</td>
        <td align="center">0.095</td>
    </tr>
    <tr>
        <td>MODEL SPIDER (Ours)</td>
        <td align="center">0.506</td>
        <td align="center"><b>0.761</b></td>
        <td align="center"><b>0.785</b></td>
        <td align="center"><b>0.909</b></td>
        <td align="center"><b>1.000</b></td>
        <td align="center"><b>0.695</b></td>
        <td align="center"><b>0.788</b></td>
        <td align="center"><b>0.954</b></td>
        <td align="center"><b>0.800</b></td>
    </tr>
    <line>
</table>

All pre-trained models were downloaded from huggingface. For more experimental results (detailed model performance on more benchmark datasets) and details, please refer to our [paper](https://arxiv.org/abs/2306.03900).

&nbsp;

## Code Implementation

### Prerequisites

&emsp; Please refer to [`requirements.txt`](requirements.txt).


### Quick Start & Reproduce

- Choose your path **/xxx/xx** to store data:
    ```bash
    source ./scripts/modify-var.sh /xxx/xx
    ```

- Download the pre-trained spider [this location](TODO) into **/xxx/xx**, and run the following command:

    ```shell
    bash scripts/run-learnware-trainer-test.sh /xxx/xx/pre_trained.pth
    ```
    The results will be displayed on the screen.

&nbsp;

## Reproduce for Other Baselines
Some benchmarking methods can be time-consuming. We have already provided the results in the benchmark.txt file. To run the benchmarks, use the following command:
```shell
bash scripts/run-baseline.sh
```
The original scores will be saved in benchmark.txt.

&nbsp;

## Contributing

Model Spider is currently in active development, and we warmly welcome any contributions aimed at enhancing capabilities. Whether you have insights to share regarding pre-trained models, data, or innovative ranking methods, we eagerly invite you to join us in making Model Spider even better.

&nbsp;

## Citing Model Spider

```latex
@article{model-spider-abs-2306-03900,
  author = {Yi{-}Kai Zhang and
            Ting{-}Ji Huang and
            Yao{-}Xiang Ding and
            De{-}Chuan Zhan and
            Han{-}Jia Ye},
  title = {Model Spider: Learning to Rank Pre-Trained Models Efficiently},
  journal = {CoRR},
  volume = {abs/2306.03900},
  year = {2023},
  url = {https://doi.org/10.48550/arXiv.2306.03900},
  doi = {10.48550/ARXIV.2306.03900},
  eprinttype = {arXiv},
  eprint = {2306.03900}
}

@misc{ModelSpider2023,
  author = {Model Spider Contributors},
  title = {Model-Spider},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zhangyikaii/Model-Spider}}
}
```
