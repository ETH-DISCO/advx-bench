we want to improve vision-based turing-tests using advx.

# 1. find the strongest models

we're looking for the strongest zero shot models as of july 2024.

they must be open source, easy to use and fine-tune.

## papers with code benchmarks

> see: https://paperswithcode.com/area/computer-vision
>
> most comprehensive

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

zero shot classification:

- https://paperswithcode.com/task/zero-shot-transfer-image-classification
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-6 → LiT-22b (2023), LiT ViT-e (2022), CoCa (2022), EVA-CLIP-18B (2024)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-4 → CoCa (2022), LiT ViT-e (2022), LiT-22B (2023), EVA-CLIP-18B (2024)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-5 → CoCa (2022), LiT-22B (2023), LiT ViT-e (2022), EVA-CLIP-18B (2024)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-3 → LiT-22B (2023), CoCa (2022), LiT ViT-e (2022), LiT-tuning (2021)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-1 → M2-Encoder (2024), CoCa (2022), LiT-22B (2023), LiT ViT-e (2022)

    best models: EVA-CLIP-18B (2024), LiT-22B (2023), CoCa (2022)

    most popular dataset: ImageNet

zero shot segmentation:

- https://paperswithcode.com/task/zero-shot-segmentation
    - https://paperswithcode.com/sota/zero-shot-segmentation-on-segmentation-in-the → Grounded HQ-SAM (2023), Grounded-SAM (2023)
    - https://paperswithcode.com/sota/zero-shot-segmentation-on-ade20k-training → GEM MetaCLIP (2023)
    
    best models: Grounded HQ-SAM (2023), Grounded-SAM (2023), GEM MetaCLIP (2023)

    most popular dataset: segmentation-in-the-wild

zero shot object detection:

- https://paperswithcode.com/task/zero-shot-object-detection
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-ms-coco → SeeDS (2023), ZSD-SCR (2022), ZSD-RRFS (2022)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-pascal-voc-07 → SeeDS (2023), ZSD-RRFS (2022)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0 → Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0-val → Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw → Grounding DINO 1.5 Pro (2024)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco → Grounding DINO 1.5 Pro (2024)

    best models: Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023), SeeDS (2023)

    most popular dataset: MS-COCO 

## huggingface trends

> see: https://huggingface.co/models
> 
> popularity doesn't always mean quality

- zero shot classification: openai/clip-vit-large-patch14, google/siglip-so400m-patch14-384
- segmentation: cidas/clipseg-rd64-refined
- zero shot dectection: google/owlv2-base-patch16-ensemble, idea-research/grounding-dino-tiny

## github trends

> see: https://roboflow.com/models / https://ossinsight.io/collections/artificial-intelligence
> 
> popularity doesn't always mean quality

- classification: yolov9, openai clip, google vit, google siglip, meta clip, resnet32
- segmentation: nvidia segformer
- detection: yolov9, grounding dino, meta detectron2, google mediapipe, meta detr

## goldblum et al. (2023)

> see: https://arxiv.org/pdf/2310.19909 (see chapter: "4.1 Task-Specific Backbones")
> 
> great benchmarking, but not zero-shot

- classification:
    1. Supervised SwinV2-Base trained on IN-21k (finetuned on IN-1k)
    2. CLIP ViT-Base trained on IN-21k
    3. Supervised ConvNeXt-Base trained on IN-21k
- segmentation and detection (same results):
    1. Supervised ConvNeXt-Base trained on IN-21K
    2. Supervised SwinV2-Base trained on IN-21k (finetuned on IN-1k)
    3. Supervised ConvNeXt-Base trained on IN-1k

## xu et al. (2023)

> see: https://www.amazon.science/publications/benchmarking-zero-shot-recognition-with-vision-language-models-challenges-on-granularity-and-specificity / https://assets.amazon.science/cb/e3/e85cc0ca4eb2a81cb223e973ae6e/benchmarking-zero-shot-recognition-with-vision-language-models-challenges-on-granularity-and-specificity.pdf
> 
> rejected from ICLR: https://openreview.net/forum?id=hdYqGkSr9S&referrer=%5Bthe%20profile%20of%20Zhenlin%20Xu%5D(%2Fprofile%3Fid%3D~Zhenlin_Xu1)
> 
> great benchmarking, but not zero-shot

- classification:
- segmentation:
- detection:

## wang et al. (2024) 🔥

very recent, uses just the right metrics and datasets.

see: https://arxiv.org/pdf/2403.10499





## excluded benchmarks

*excluded benchmarks (outdated or poor quality):*

- https://huggingface.co/spaces/hf-vision/object_detection_leaderboard (very limited)
- https://cocodataset.org/#detection-leaderboard (outdated, from 2020)
- https://segmentmeifyoucan.com/leaderboard / https://arxiv.org/pdf/2104.14812 (outdated, from 2021)












# 2. evaluate them against a variety of advx

https://arxiv.org/pdf/2403.10499 🔥 does exactly what we're doing


see: https://github.com/wang-research-lab/roz/blob/main/download_cifar.py / https://github.com/wang-research-lab/roz/blob/main/scripts/common_adversarial_attack/run_common_adversarial_attack.py

https://timm.fast.ai/

generating advx:

- https://paperswithcode.com/task/adversarial-attack
- https://paperswithcode.com/task/real-world-adversarial-attack
- https://paperswithcode.com/task/adversarial-attack-detection
- see: http://videos.rennes.inria.fr/seminaire-SoSySec/Maura-Pintor-03-02-2023/20230203-Maura-Pintor-sosysec-slides.pdf
- see: `torchattack`

*datasets:*

- aggregators:
    - https://datasetninja.com/
    - https://huggingface.co/datasets
- captchas:
    - https://github.com/orlov-ai/hcaptcha-dataset
    - https://github.com/Inefficacy/Captcha-Datasets
    - https://www.kaggle.com/datasets/mikhailma/test-dataset
    - https://www.kaggle.com/datasets/cry2003/google-recaptcha-v2-images
    - https://datasetninja.com/google-recaptcha-image
- self-made:
    - https://github.com/hendrycks/natural-adv-examples
    - try rotating images
    - try to integrate things that are invisible to humans but visible to models (ie. low transparency, undersaturated values)
    - try out new hcaptcha dataset with ascii characters and funny colors (reach out to turlan)
        - https://gitlab.ethz.ch/disco-students/fs24/image-captchas
        - https://gitlab.ethz.ch/disco-students/fs24/image-captchas/-/blob/main/assets/datasets/hcaptcha_dataset_turlan/processed/0a28e4fc9452e9b90e2e08e564de91754b85db04dba33f7216378aac883f5d3f.png?ref_type=heads

new exercises:

- try selecting a subset of images ("only select 2 horses of 4")
