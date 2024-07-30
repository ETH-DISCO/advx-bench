we want to improve vision-based turing-tests using advx.

we're looking for the strongest zero shot image recognition models (umbrella term) as of july 2024.

they must be:

- open source
- easy to use and fine-tune
- have a good balance between efficiency and effectiveness
- able to run on consumer hardware (ie. standard standing desk pc with gpu)

these are our findings.

# 1. performance benchmarks

overview:

- huggingface: https://huggingface.co/docs/transformers/en/model_doc/beit
- timm: https://huggingface.co/docs/timm/en/models
- course: https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/pre-intro

## 1.1. RoZ benchmark (CVPR 2024) 🔥

> see: https://arxiv.org/pdf/2403.10499
>
> has the same goal as us - but only focuses on **CLIP models** for cls tasks on ImageNet

ranking by performance (based on table 2)

- **ViT-B/16** (Standard ImageNet model) - 84.20% accuracy
- **ResNet50x16** (CLIP model) - 70.67% accuracy
- ViT-B/32 (Standard ImageNet model)
- ResNet50x4 (CLIP model)
- ResNet101 (Standard ImageNet model)

ranking by adversarial robustness (based on table 1, figure 8)

- **ViT-B/16** (Standard ImageNet model)
- ViT-B/32 (Standard ImageNet model)
- ResNet101 (Standard ImageNet model)
- ResNet50 (Standard ImageNet model)
- ResNet50x16 (CLIP model)

## 1.2. pwc benchmarks

> see: https://paperswithcode.com/area/computer-vision
>
> most comprehensive collection of models

classification:

- https://paperswithcode.com/area/computer-vision/image-classification (overview)
- https://paperswithcode.com/task/image-classification
- https://paperswithcode.com/task/self-supervised-image-classification
- https://paperswithcode.com/task/unsupervised-image-classification
- https://paperswithcode.com/task/efficient-vits

segmentation:

- https://paperswithcode.com/area/computer-vision/semantic-segmentation (overview)
- https://paperswithcode.com/area/computer-vision/2d-semantic-segmentation (overview)
- https://paperswithcode.com/task/image-segmentation
- https://paperswithcode.com/task/semantic-segmentation
- https://paperswithcode.com/task/universal-segmentation (no data)

object detection:

- https://paperswithcode.com/area/computer-vision/object-detection
- https://paperswithcode.com/area/computer-vision/2d-object-detection

---

zero shot classification:

- https://paperswithcode.com/task/zero-shot-transfer-image-classification
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-6 → **LiT-22b (2023)**, LiT ViT-e (2022), **CoCa (2022)**, **EVA-CLIP-18B (2024)**
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-4 → CoCa (2022), LiT ViT-e (2022), LiT-22B (2023), EVA-CLIP-18B (2024)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-5 → CoCa (2022), LiT-22B (2023), LiT ViT-e (2022), EVA-CLIP-18B (2024)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-3 → LiT-22B (2023), CoCa (2022), LiT ViT-e (2022), LiT-tuning (2021)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-1 → M2-Encoder (2024), CoCa (2022), LiT-22B (2023), LiT ViT-e (2022)

zero shot segmentation:

- https://paperswithcode.com/task/zero-shot-segmentation
    - https://paperswithcode.com/sota/zero-shot-segmentation-on-segmentation-in-the → **Grounded HQ-SAM (2023)**, **Grounded-SAM (2023)**
    - https://paperswithcode.com/sota/zero-shot-segmentation-on-ade20k-training → **GEM MetaCLIP (2023)**

zero shot object detection:

- https://paperswithcode.com/task/zero-shot-object-detection
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-ms-coco → SeeDS (2023), ZSD-SCR (2022), ZSD-RRFS (2022)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-pascal-voc-07 → SeeDS (2023), ZSD-RRFS (2022)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0 → **Grounding DINO 1.5 Pro (2024)**, **OWLv2 (2023)**, **MQ-GLIP-L (2023)**
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0-val → Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw → Grounding DINO 1.5 Pro (2024)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco → Grounding DINO 1.5 Pro (2024)

## 1.3. goldblum et al. (2023)

> see: https://arxiv.org/pdf/2310.19909 (see chapter: "4.1 Task-Specific Backbones")
> 
> very thoughtful benchmarking based on the most significant metrics, but only focused on backbones (feature extraction)

- classification:
    - Supervised SwinV2-Base trained on IN-21k (finetuned on IN-1k)
    - CLIP ViT-Base trained on IN-21k
    - Supervised ConvNeXt-Base trained on IN-21k
- segmentation and detection (same results):
    - Supervised ConvNeXt-Base trained on IN-21K
    - Supervised SwinV2-Base trained on IN-21k (finetuned on IN-1k)
    - Supervised ConvNeXt-Base trained on IN-1k

## 1.4. huggingface trends

> see: https://huggingface.co/models
> 
> popularity isn't a good metric

- zero shot classification: openai/clip-vit-large-patch14, google/siglip-so400m-patch14-384
- segmentation: cidas/clipseg-rd64-refined
- zero shot dectection: google/owlv2-base-patch16-ensemble, idea-research/grounding-dino-tiny

## 1.5 github trends

> see: https://roboflow.com/models / https://ossinsight.io/collections/artificial-intelligence
> 
> popularity isn't a good metric

- classification: yolov9, openai clip, google vit, google siglip, meta clip, resnet32
- segmentation: nvidia segformer
- detection: yolov9, grounding dino, meta detectron2, google mediapipe, meta detr

## 1.6 pytorch image models benchmarks

> see: https://github.com/huggingface/pytorch-image-models
> 
> analytics: https://www.kaggle.com/code/jhoward/which-image-models-are-best
>
> not comprehensive, not zero-shot, but a very good quantitative benchmark

top1 accuracy by family:

- beit (88.60% top1, by beit_large_patch16_512.in22k_ft_in22k_in1k)
- convnext (87.47% top1, by convnext_large.fb_in22k_ft_in1k_384)
- swin (87.13% top1, by swin_large_patch4_window12_384.ms_in22k_ft_in1k)
- efficientnetv2 (84.81% top1, by efficientnetv2_rw_m.agc_in1k)
- resnetd (83.96% top1, by resnet200d.ra2_in1k)
- resnet (83.45% top1, by resnet152.a1h_in1k)
- regnetx (82.81% top1, by regnetx_320.tv2_in1k)
- levit (82.60% top1, by levit_384.fb_dist_in1k)
- vgg (74.22% top1, by vgg19_bn.tv_in1k)

## 1.7. other benchmarks worth mentioning

these benchmarks are outdated or not comprehensive enough:

- https://www.amazon.science/publications/benchmarking-zero-shot-recognition-with-vision-language-models-challenges-on-granularity-and-specificity / https://assets.amazon.science/cb/e3/e85cc0ca4eb2a81cb223e973ae6e/benchmarking-zero-shot-recognition-with-vision-language-models-challenges-on-granularity-and-specificity.pdf) (great paper, but not a benchmark but critique of existing metrics)
- https://huggingface.co/spaces/hf-vision/image_classification_leaderboard (only object detection, very limited)
- https://cocodataset.org/#detection-leaderboard (outdated, from 2020)
- https://segmentmeifyoucan.com/leaderboard / https://arxiv.org/pdf/2104.14812 (outdated, from 2021)

# 2. ease of use

## 2.1. cls

google lit @ 2021:

- https://arxiv.org/abs/2111.07991
- https://github.com/google-research/vision_transformer
- https://github.com/google-research/vision_transformer/blob/main/model_cards/lit.md#model-data
- https://github.com/google-research/vision_transformer?tab=readme-ov-file#lit-models
- based on clip, but beats it in zero-shot classification
- ✅ installation:
    - https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb

eva-clip-18b @ 2023:

- https://arxiv.org/pdf/2402.04252
- https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B
- clip with 18 billion parameters
- the largest model is 35.3GB large, trained using apex and xformer
- ✅ installation:
    - https://huggingface.co/papers/2402.04252
    - https://huggingface.co/models?search=evaclip
    - https://huggingface.co/BAAI/EVA-CLIP-8B (should be small enough to run on a consumer laptop)
    - https://huggingface.co/BAAI/EVA-CLIP-18B
    - https://huggingface.co/BAAI/EVA-CLIP-8B-448

google efficientnet @ 2020:

- https://arxiv.org/pdf/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- ✅ installation:
    - https://huggingface.co/google/efficientnet-b0
    - https://huggingface.co/docs/timm/en/models/efficientnet

openai clip @ 2021:

- https://arxiv.org/abs/2103.00020v1
- https://openai.com/index/clip/
- https://github.com/openai/CLIP
- most influential and widely-used models for zero-shot image classification
- same performance as ResNet50 on ImageNet zero-shot classification
- uses a vision transformer (ViT) as the image encoder and a text encoder based on the transformer architecture
- models:
    - https://github.com/openai/CLIP/blob/main/model-card.md
    - `clip.available_models()`: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
- ✅ installation:
    - https://huggingface.co/mlunar/clip-variants/raw/555f7ba437324dd8e06b4e73fbd1605e6a0ba753/convert.py
    - https://github.com/wang-research-lab/roz/blob/main/scripts/natural_distribution_shift/src/models/CLIPViT.py
    - https://huggingface.co/openai/clip-vit-large-patch14
    - also see implementation by RoZ benchmark paper

open-clip + open-coca @ 2022 :

- https://arxiv.org/abs/2212.07143
- coca: https://arxiv.org/abs/2205.01917, https://github.com/lucidrains/CoCa-pytorch
- ✅ installation:
    - https://github.com/mlfoundations/open_clip?tab=readme-ov-file#usage

clip-dinoiser @ 2023:

- https://arxiv.org/abs/2312.12359
- adds DINO for open vocabulary semantic segmentation
- not part of the benchmark leaderboard, just a cool idea
- ✅ installation:
    - https://github.com/wysoczanska/clip_dinoiser/

## 2.2. det

grounding dino @ 2024:

- https://arxiv.org/abs/2303.05499
- https://github.com/IDEA-Research/GroundingDINO
- dino is an improvement over DETR by meta from 2022: https://arxiv.org/abs/2203.03605, https://arxiv.org/abs/2005.12872
- bleeding edge, still actively being developed
- ✅ installation:
    - https://huggingface.co/IDEA-Research/grounding-dino-base
    - https://huggingface.co/IDEA-Research/grounding-dino-tiny

owlv2 / owlvit @ 2023:

- https://arxiv.org/abs/2305.01917
- ✅ installation:
    - https://huggingface.co/google/owlv2-base-patch16-ensemble → uses clip as backbone with vit-B/16
    - https://huggingface.co/google/owlvit-base-patch32 → uses clip as backbone with vit-B/32 (slower but more accurate)

mq-det @ 2023:

- https://arxiv.org/pdf/2305.18980v2
- https://github.com/yifanxu74/mq-det
- ❌ installation:
    - instructions unclear, environment not reproducible, weights have to be downloaded manually

grounding dino 1.5 @ 2024:

- https://arxiv.org/abs/2303.05499
- https://github.com/IDEA-Research/Grounding-DINO-1.5-API
- bleeding edge, still actively being developed
- ❌ installation:
    - only through paid api https://deepdataspace.com/request_api, not open source

## 2.3. seg

> these are object segmentation models, not semantic segmentation models which makes them less useful for our purposes
>
> i therefore decided to leave them out from our evaluation pipeline

gem metaclip @ 2024:

- https://arxiv.org/pdf/2309.16671
- https://github.com/facebookresearch/MetaCLIP
- beats clip and openclip by a wide margin in zero-shot classification
- massive speedup possible through flash attention and scaled dot product attention (but requires a gpu): https://huggingface.co/docs/transformers/main/en/model_doc/clip#expected-speedups-with-flash-attention-and-sdpa
- ✅ installation:
    - https://github.com/facebookresearch/MetaCLIP?tab=readme-ov-file#quick-start
    - https://huggingface.co/models?search=metaclip
    - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b (largest version)

sam vit @ 2023:

- https://arxiv.org/pdf/2304.02643
- https://github.com/facebookresearch/segment-anything
- ✅ installation:
    - https://huggingface.co/facebook/sam-vit-base
    - https://huggingface.co/facebook/sam-vit-huge (largest version)
    - https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
    - https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

grounded sam @ 2024:

- https://arxiv.org/abs/2401.14159
- https://github.com/IDEA-Research/Grounded-Segment-Anything
- https://huggingface.co/spaces/linfanluntan/Grounded-SAM
- ❌ installation:
    - given that this is the only model that must be run in a docker container, it's not worth the effort to orchestrate multiple containers
    - docker doesn't support [apple mps](https://github.com/pytorch/pytorch/issues/81224) as a pytorch backend which then restricts the entire project to cpu or cuda backends

grounded hq-sam @ 2023:

- https://arxiv.org/abs/2306.01567
- https://github.com/SysCV/sam-hq
- ❌ installation:
    - model checkpoint must be downloaded from sketchy google drive link
    - only the smallest version (sub 50MB can be pushed to git) can be pushed to git with significantly reduced performance
