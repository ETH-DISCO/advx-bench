we want to improve vision-based turing-tests using advx.

we're looking for the strongest zero shot models as of july 2024.

they must be open source, easy to use and fine-tune.



# wang et al. (2024) benchmarks 🔥

very recent, uses just the right metrics and datasets.

see: https://arxiv.org/pdf/2403.10499

...






# pwc benchmarks

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

# excluded benchmarks

*goldblum et al. (2023):*

> see: https://arxiv.org/pdf/2310.19909 (see chapter: "4.1 Task-Specific Backbones")
> 
> very thoughtful benchmarking based on the most significant metrics, but only focuses on backbones (feature extraction)

- classification:
    1. Supervised SwinV2-Base trained on IN-21k (finetuned on IN-1k)
    2. CLIP ViT-Base trained on IN-21k
    3. Supervised ConvNeXt-Base trained on IN-21k
- segmentation and detection (same results):
    1. Supervised ConvNeXt-Base trained on IN-21K
    2. Supervised SwinV2-Base trained on IN-21k (finetuned on IN-1k)
    3. Supervised ConvNeXt-Base trained on IN-1k

*huggingface trends:*

> see: https://huggingface.co/models
> 
> popularity isn't a good metric

- zero shot classification: openai/clip-vit-large-patch14, google/siglip-so400m-patch14-384
- segmentation: cidas/clipseg-rd64-refined
- zero shot dectection: google/owlv2-base-patch16-ensemble, idea-research/grounding-dino-tiny

*github trends:*

> see: https://roboflow.com/models / https://ossinsight.io/collections/artificial-intelligence
> 
> popularity isn't a good metric

- classification: yolov9, openai clip, google vit, google siglip, meta clip, resnet32
- segmentation: nvidia segformer
- detection: yolov9, grounding dino, meta detectron2, google mediapipe, meta detr

*others:*

- https://www.amazon.science/publications/benchmarking-zero-shot-recognition-with-vision-language-models-challenges-on-granularity-and-specificity / https://assets.amazon.science/cb/e3/e85cc0ca4eb2a81cb223e973ae6e/benchmarking-zero-shot-recognition-with-vision-language-models-challenges-on-granularity-and-specificity.pdf) (great paper, zero shot but for vision-language models, not vision-only)
- https://huggingface.co/spaces/hf-vision/image_classification_leaderboard (only object detection, very limited)
- https://cocodataset.org/#detection-leaderboard (outdated, from 2020)
- https://segmentmeifyoucan.com/leaderboard / https://arxiv.org/pdf/2104.14812 (outdated, from 2021)
