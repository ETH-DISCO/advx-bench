we want to improve vision-based turing-tests using advx.

we're looking for the strongest zero shot models as of july 2024.

they must be open source, easy to use and fine-tune.

these are our findings.

# 1. model selection

overview:

- huggingface: https://huggingface.co/docs/transformers/en/model_doc/beit
- timm: https://huggingface.co/docs/timm/en/models

## 1.1. RoZ benchmark (CVPR 2024) 🔥

> see: https://arxiv.org/pdf/2403.10499
>
> has the same goal as us - but only focuses on **CLIP models** for cls tasks on ImageNet

ranking by performance (based on table 2)

- **ViT-B/16** (Standard ImageNet model) - 84.20% accuracy 🔥
- **ResNet50x16** (CLIP model) - 70.67% accuracy
- ViT-B/32 (Standard ImageNet model)
- ResNet50x4 (CLIP model)
- ResNet101 (Standard ImageNet model)

ranking by adversarial robustness (based on table 1, figure 8)

- **ViT-B/16** (Standard ImageNet model) 🔥
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

# 2. conclusion

some of the most recent breakthroughs in zero shot image recognition (this umbrella term includes cls, seg, det) are:

## 2.1. best models for classification

google lit @ 2021:

- https://arxiv.org/abs/2111.07991
- https://github.com/google-research/vision_transformer
- https://github.com/google-research/vision_transformer/blob/main/model_cards/lit.md#model-data
- based on clip, but beats it in zero-shot classification

coca @ 2022:

- https://arxiv.org/abs/2205.01917
- https://github.com/lucidrains/CoCa-pytorch

eva-clip-18b @ 2023:

- https://arxiv.org/pdf/2402.04252
- https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B
- https://huggingface.co/papers/2402.04252
- clip with 18 billion parameters
- ❌ 35.3GB model size, based on apex and xformer for distributed training

openai clip @ 2021:

- https://arxiv.org/abs/2103.00020v1
- https://openai.com/index/clip/
- https://github.com/openai/CLIP
- https://github.com/openai/CLIP/blob/main/model-card.md
- https://deepgram.com/ai-glossary/zero-shot-classification-models
- https://huggingface.co/mlunar/clip-variants/raw/555f7ba437324dd8e06b4e73fbd1605e6a0ba753/convert.py
- https://github.com/wang-research-lab/roz/blob/main/scripts/natural_distribution_shift/src/models/CLIPViT.py
- most influential and widely-used models for zero-shot image classification
- same performance as ResNet50 on ImageNet zero-shot classification
- uses a vision transformer (ViT) as the image encoder and a text encoder based on the transformer architecture
- implementations: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'] (run `clip.available_models()` to see all)
- variations:
    - https://github.com/mlfoundations/open_clip → also handles CoCa
    - https://github.com/wysoczanska/clip_dinoiser/ → adds DINO for open vocabulary semantic segmentation

google efficientnet @ 2020:

- https://arxiv.org/pdf/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://huggingface.co/google/efficientnet-b0
- https://huggingface.co/docs/timm/en/models/efficientnet

## 2.2. best models for segmentation

gem metaclip @ 2024:

- https://arxiv.org/pdf/2309.16671
- https://github.com/facebookresearch/MetaCLIP
- beats clip and openclip by a wide margin in zero-shot segmentation

grounded hq-sam @ 2023:

- https://arxiv.org/abs/2306.01567
- https://github.com/SysCV/sam-hq
- ❌ model checkpoint must be downloaded from sketchy google drive link, only works with cuda

grounded sam @ 2024:

- https://arxiv.org/abs/2401.14159
- https://github.com/IDEA-Research/Grounded-Segment-Anything
- comes with docker container, can be managed with docker-compose

sam @ 2023:

- https://arxiv.org/pdf/2304.02643
- https://github.com/facebookresearch/segment-anything
- ✅ very straightforward

## 2.3. best models for object detection

grounding dino @ 2024:

- https://arxiv.org/abs/2303.05499
- https://github.com/IDEA-Research/GroundingDINO
- https://github.com/IDEA-Research/Grounding-DINO-1.5-API (improved)
- DINO is an improvement over DETR by meta from 2022: https://arxiv.org/abs/2203.03605, https://arxiv.org/abs/2005.12872
-  ❌ not open source - you have to pay for their api key from https://deepdataspace.com/request_api

owlv2 @ 2023:

- https://arxiv.org/abs/2305.01917
- https://huggingface.co/google/owlv2-base-patch16-ensemble → uses clip as backbone with vit-B/16
- ✅ very straightforward to use, small footprint

mq-det @ 2023:

- https://arxiv.org/pdf/2305.18980v2
- https://github.com/yifanxu74/mq-det
- ❌ installation process unclear
