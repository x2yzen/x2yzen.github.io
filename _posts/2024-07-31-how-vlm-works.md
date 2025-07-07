---
title: How VLM Works
date: 2024-07-31 08:00:00 +0800
categories: [language_model]
tags: [vlm]     # TAG names should always be lowercase
pin: false
math: true
---

Tracing the evolutionary journey of vision models.

First, let's establish clear definitions: Vision-language models (VLMs) that take images and texts as inputs and output texts. Therefore, various text-to-image models are excluded here—our focus is primarily on image understanding (from simple classification to complex ad-hoc QA).

![](/assets/images/2024-07-31-how-vlm-works/image.png)

**TL;DR**

| Representative Model        | Main Contribution                                                  | Capability                    |
| --------------------------- | ------------------------------------------------------------------ | ----------------------------- |
| ConvNets (1998)            | Ancient vision models                                              | Image classification (preset categories) |
| ViT (2021)                 | Used language models (TF encoder) to match ConvNets in vision     | Image classification (preset categories) |
| CLIP (2021)                | Proposed image encoding (ViT) + text encoding + connector transfer learning paradigm (cos similarity matrix) | Image classification (arbitrary categories) |
| KOSMOS, QWEN-VL etc. (2023-) | SOTA VLM, using TF decoder instead of cos similarity matrix as connector | Arbitrary Q&A               |

## ConvNets

Ancient vision models, tracing back to *Yann LeCun in 1998* (http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

Mainly composed of four components, capable of image classification tasks:

![](/assets/images/2024-07-31-how-vlm-works/conv.png)![](/assets/images/2024-07-31-how-vlm-works/image-1.png)![](/assets/images/2024-07-31-how-vlm-works/image-2.png)

Many [blogs](https://medium.com/neuronio/understanding-convnets-cnn-712f2afe4dd3) explain this clearly. Since the principles have weak correlation with SOTA VLMs, I won't elaborate. In essence, models at this stage could **classify images into categories seen during training, given an input image.**

## ViT

*Google @ICLR 2021*

https://arxiv.org/abs/2010.11929

https://huggingface.co/docs/transformers/en/model_doc/vit

Compared to ConvNets, the main contribution is discovering that **Transformers can achieve nearly identical efficiency as ConvNets in image feature extraction**, essentially replacing various kernels in ConvNets with self-attention blocks. Considering Transformers' outstanding performance in natural language processing, ViT obviously has better generalizability than ConvNets as a feature extractor.

> *Reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks*

The architecture is remarkably simple: essentially cutting a complete image into patch sequences, mapping them through embedding layers, treating them analogously to text token sequences fed into TF encoders, encoding through self-attention mechanisms, then using the output for MLP classification.

* Input: [class] + pic_patch_seq + pos_embedding
* Arch: multi-head self-attention block (no causal mask) * L -> [class] embedding -> MLP_head -> label cross entropy loss

![](/assets/images/2024-07-31-how-vlm-works/image-3.png)

![](/assets/images/2024-07-31-how-vlm-works/image-4.png)

See this [code sample](https://github.com/BrianPulfer/PapersReimplementations/blob/main/src/cv/vit/vit_torch.py)

## CLIP

*OpenAI @ICML 2021*

https://arxiv.org/abs/2103.00020

https://huggingface.co/openai/clip-vit-large-patch14-336

Building on ViT, the main contribution lies in model architecture. **CLIP doesn't just extract features from images, but designs a paradigm of visual encoder (essentially ViT) + text encoder + connector, attempting to transfer knowledge learned by two models from images and corresponding natural language respectively.** The final effect achieved is that image classification models gained **zero-shot** generalization capability—no longer limited to categories seen during training. Given an image and arbitrary descriptive options, the model can make classification predictions. Conversely, it can also perform image search through arbitrary given text descriptions.

![](/assets/images/2024-07-31-how-vlm-works/image-5.png)

Paper abstract (could be skipped)

![](/assets/images/2024-07-31-how-vlm-works/clip.png)

The training process can be summarized as:

1. Prepare N pairs of matching images and text (captions)
2. Convert to embedding vectors through text encoder and vision encoder respectively
3. Project both embedding vectors to the same space through MLP
4. Calculate dot products pairwise (N*N)
5. Calculate classification loss based on whether images and text match (N out of N*N) and backpropagate to train encoders
6. (Prediction phase) Given a new image and several candidate text descriptions, use the same pipeline—the pair with maximum dot product is the most likely prediction for the new image

![](/assets/images/2024-07-31-how-vlm-works/filename.png)

Many subsequent works build on similar architectures, particularly exploring various forms of connectors extensively.

## KOSMOS-1 & QWEN-VL etc.

*MSFT@2023*

https://arxiv.org/abs/2302.14045

https://github.com/microsoft/unilm

*Alibaba Group @ 2023*

https://arxiv.org/abs/2308.12966

https://huggingface.co/Qwen/Qwen-VL

Using KOSMOS as an example, building on CLIP, KOSMOS's main contribution is **using TF decoder (rather than simple dot product matrix) as fusion connector, using attention mechanisms (rather than cos similarity) to mutually transfer knowledge between images and text.** In terms of usage, since the connector becomes an autoregressive generative model, KOSMOS transcends the classifier paradigm and can support various ad-hoc QAs.

![](/assets/images/2024-07-31-how-vlm-works/image-6.png)

KOSMOS uses interleaved image-text data (not just image-text pairs) as training corpus. After separately embedding images and text, they're interspersed together, then TF decoder processes multi-modal embedding tokens in a standard & unified manner (i.e., causal-mask -> self attention block*N -> MLP -> predict-the-next token. Note: embedding tokens of modalities other than text are not included in the loss function, meaning the model won't target generating image modalities—output can only be text modality).

Since this paper (surprisingly) lacks diagrams, I drew a simplified architecture diagram (omitting details like feature resampler). See this [code sample](https://github.com/bjoernpl/KOSMOS_reimplementation/blob/main/kosmos.py).

![](/assets/images/2024-07-31-how-vlm-works/k1.png)

Paper abstract (could be skipped):

![](/assets/images/2024-07-31-how-vlm-works/k2.png)
![](/assets/images/2024-07-31-how-vlm-works/k3.png)
![](/assets/images/2024-07-31-how-vlm-works/image-7.png)

Until now, the overall architecture of VLM models hasn't changed significantly (for instance, QWEN-VL's biggest change is using QWEN-7B instead of MAGNETO). Most work involves fine-tuning the above structure (adding feature sampling/compression, layer norm position adjustments, etc.).

Building increasingly large training datasets, exploring optimal data ratios, and researching how to develop image understanding and language generation capabilities independently without interference (challenging, since they share a language foundation) have become key to improving model performance (and simultaneously the latest mystical subjects).

***

## Others

### Idefics2

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