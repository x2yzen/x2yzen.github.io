---
title: How VLM Works
date: 2024-07-31 08:00:00 +0800
categories: [llm]
tags: [vlm]     # TAG names should always be lowercase
pin: true
math: true
---
梳理视觉模型的发展历程。

首先明确定义：Vision-language models (VLMs) that take images and texts as inputs and output texts，因此各种文生图的模型不在此列，主要用于图片理解（从最简单的分类到复杂的adhoc QA）。

![](/assets/images/2024-07-31-how-vlm-works/image.png)

**TL;DR**

| 代表模型                        | 主要贡献                                                        | 能力         |
| --------------------------- | ----------------------------------------------------------- | ---------- |
| convnets(1998)              | 上古视觉模型                                                      | 图片分类（预设种类） |
| ViT(2021)                   | 用语言模型（TF encoder）在视觉领域打平convnets                            | 图片分类（预设种类） |
| CLIP(2021)                  | 提出图片编码（ViT）+ 文字编码 + connector迁移学习（cos similarity matrix）的范式 | 图片分类（任意种类） |
| KOSMOS、QWEN-VL etc. (2023-) | SOTA VLM，使用TF decoder代替cos similarity matrix，行使connector的职责 | 任意问答       |

## ConvNets

古老的视觉模型，追溯到 [*Yann LeCun in 1998*](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

主要由四个部分组成，可以进行图片分类等工作

![](/assets/images/2024-07-31-how-vlm-works/conv.png)![](/assets/images/2024-07-31-how-vlm-works/image-1.png)![](/assets/images/2024-07-31-how-vlm-works/image-2.png)

很多[blog](https://medium.com/neuronio/understanding-convnets-cnn-712f2afe4dd3)讲得很清晰，原理和SOTA的VLM关系较弱，就不赘述了。总之这个阶段模型的效果是**给定一张图片，对训练中见过的category进行分类。**

## ViT

*Google @ICLR 2021*

https://arxiv.org/abs/2010.11929

https://huggingface.co/docs/transformers/en/model\_doc/vit

对比convnets，主要的贡献是发现**TF在图片特征提取上可以和ConvNets在达到几乎一样的效率**，本质上是使用self-attention block代替了convnets中的各种kernel。考虑到TF在自然语言处理中的突出效果，作为特征提取器，ViT显然比ConvNets具备更好的泛用性

> *Reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks *

架构确实相当简单：其实就是把一张完整的图片切割为patch sequence，通过embedding layer的映射，类比为文字的token sequence塞进TF encoder，通过自注意力机制编码以后把产物拿来过MLP分类

* Input: \[class] + pic\_patch\_seq + pos\_embedding

* Arch: multi-head self-attention block (no causal mask) \* L -> \[class] embedding -> MLP\_head -> label cross entropy loss

![](/assets/images/2024-07-31-how-vlm-works/image-3.png)

![](/assets/images/2024-07-31-how-vlm-works/image-4.png)

看一个[code sample](https://github.com/BrianPulfer/PapersReimplementations/blob/main/src/cv/vit/vit_torch.py)

## CLIP

*OpenAI @ICML 2021*

https://arxiv.org/abs/2103.00020

https://huggingface.co/openai/clip-vit-large-patch14-336

建立在ViT之上，主要的贡献是在模型架构上，**CLIP不仅对图片进行特征提取，而是设计了一个visual encoder（实际上就是ViT） + text encoder + connector的paradigm，尝试transfer两个模型分别从图片和对应的自然语言中学习到的知识，**最终达成的效果是，图片分类模型获得了**zero-shot** 泛化能力，i.e. 不再局限于训练时见过的category，给定图片和任意几个描述选项，模型可以进行分类预测；反过来，也可以通过任意给定的文字描述进行image search。

![](/assets/images/2024-07-31-how-vlm-works/image-5.png)

paper abstract (could be skipped)

![](/assets/images/2024-07-31-how-vlm-works/clip.png)

训练过程简单概括为：

1. 准备匹配的图片和文字（caption）N对

2. 分别过text encoder和vision encoder转换为embedding vector

3. 将两份embedding vector通过MLP投射到同一空间

4. 两两计算点积（N\*N）

5. 按照图片与文字匹配与否（N out of N\*N）计算分类loss并反向传播，训练encoder

6. （预测阶段）给定一张新的图片，并写出几个备选的文字描述，通过同样的流水线，点积最大的那个pair就是对新图片的最可能的预测描述

![](/assets/images/2024-07-31-how-vlm-works/filename.png)

后续很多工作都是建立在类似的架构上，特别是充分尝试各种形式的connector。

## KOSMOS-1 & QWEN-VL etc.

*MSFT@2023*

https://arxiv.org/abs/2302.14045

https://github.com/microsoft/unilm

*Alibaba Group @ 2023*

https://arxiv.org/abs/2308.12966

https://huggingface.co/Qwen/Qwen-VL

以KOSMOS为例，在CLIP的基础上，KOSMOS主要的贡献是在**使用TF decoder（而不是简单的dot product matrix）作为fusion connector，使用attention机制（而不是cos similarity）来互相transfer图片和文字中的知识。**从使用效果上，由于connector变成了一个自回归的生成模型，KOSMOS跳出了分类器的范畴，可以支持各种各样的adhoc-QA。

![](/assets/images/2024-07-31-how-vlm-works/image-6.png)

KOSMOS使用图文混排数据（而不只是图文pair数据）作为训练语料，对其中的image和text分别embedding之后再次穿插在一起，随后TF decoder使用标准的&统一的方式处理多种模态的embedding token（i.e. causal-mask -> self attention block\*N -> MLP -> predict-the-next token，note. 除了文字以外的其它模态的embedding token不会被计入loss function，这意味着模型不会以生成图片模态作为目标，output只能有文字模态）。

由于这篇文章（竟然）没有绘图，因此自己画了一个架构简图（省略了一些feature resampler之类的细节）。看一个[code sample](https://github.com/bjoernpl/KOSMOS_reimplementation/blob/main/kosmos.py)。
![](/assets/images/2024-07-31-how-vlm-works/k1.png)

paper abstract (could be skipped):

![](/assets/images/2024-07-31-how-vlm-works/k2.png)
![](/assets/images/2024-07-31-how-vlm-works/k3.png)
![](/assets/images/2024-07-31-how-vlm-works/image-7.png)

直到现在，VLM模型的大体架构已经没有明显的变动（比如QWEN-VL最大的变动就是使用QWEN-7B代替MAGNETO），多数是在上述结构上做一些微调（加上feature sampling/compress，layer norm位置调整等等）。

构建规模越来越大的训练数据集，探索各类数据的配比，以及研究如何使图片理解和语言创作的能力各自发展，互不打架（很难，因为它们共享一个语言基座）等成为了模型效果提高的关键（也同时是最新的玄学课题）。

***
## Others

### Idefics2&#x20;

*HuggingFace @ 2024*

https://arxiv.org/abs/2405.02246

https://huggingface.co/HuggingFaceM4/idefics2-8b

### InternLM-XComposer2

https://arxiv.org/abs/2401.16420

https://huggingface.co/internlm/internlm-xcomposer2-vl-7b

![](/assets/images/2024-07-31-how-vlm-works/diagram.png)

### InternVL
https://arxiv.org/abs/2312.14238

https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5

