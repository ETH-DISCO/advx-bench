> we want to improve vision-based turing-tests using advx.

# 1. find the strongest models (as of July 2024)

> find the best models (maybe also fine tune them)

*leaderboards:*

- https://paperswithcode.com/area/computer-vision ⭐️ ← most up to date, most comprehensive
- https://segmentmeifyoucan.com/leaderboard
- https://huggingface.co/spaces/hf-vision/object_detection_leaderboard 
- https://cocodataset.org/#detection-leaderboard (outdated, from 2020)

*tasks:*

zero shot models are most relevant because we don't want to train on the specific captchas we're going to use.

- classification: image class labels
    
    - https://paperswithcode.com/area/computer-vision/image-classification (overview)
    - https://paperswithcode.com/task/image-classification
    - https://paperswithcode.com/task/self-supervised-image-classification
    - https://paperswithcode.com/task/unsupervised-image-classification
    - https://paperswithcode.com/task/efficient-vits
    - https://paperswithcode.com/task/zero-shot-transfer-image-classification ⭐️
        - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-6 → LiT-22b (2023), LiT ViT-e (2022), CoCa (2022), EVA-CLIP-18B (2024)
        - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-4 → CoCa (2022), LiT ViT-e (2022), LiT-22B (2023), EVA-CLIP-18B (2024)
        - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-5 → CoCa (2022), LiT-22B (2023), LiT ViT-e (2022), EVA-CLIP-18B (2024)
        - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-3 → LiT-22B (2023), CoCa (2022), LiT ViT-e (2022), LiT-tuning (2021)
        - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-1 → M2-Encoder (2024), CoCa (2022), LiT-22B (2023), LiT ViT-e (2022)

        best models: EVA-CLIP-18B (2024), LiT-22B (2023), CoCa (2022)

        best dataset: ImageNet

- segmentation: pixel groups (sometimes with labels)

    - https://paperswithcode.com/area/computer-vision/semantic-segmentation (overview)
    - https://paperswithcode.com/area/computer-vision/2d-semantic-segmentation (overview)
    - https://paperswithcode.com/task/image-segmentation
    - https://paperswithcode.com/task/semantic-segmentation
    - https://paperswithcode.com/task/universal-segmentation (no data)
    - https://paperswithcode.com/task/zero-shot-segmentation ⭐️
        - https://paperswithcode.com/sota/zero-shot-segmentation-on-segmentation-in-the → Grounded HQ-SAM (2023), Grounded-SAM (2023)
        - https://paperswithcode.com/sota/zero-shot-segmentation-on-ade20k-training → GEM MetaCLIP (2023)
        
        best models: Grounded HQ-SAM (2023), Grounded-SAM (2023), GEM MetaCLIP (2023)

        best dataset: segmentation-in-the-wild

- object detection: bounding boxes with labels
    
    - https://paperswithcode.com/area/computer-vision/object-detection
    - https://paperswithcode.com/area/computer-vision/2d-object-detection
    - https://paperswithcode.com/task/zero-shot-object-detection ⭐️
        - https://paperswithcode.com/sota/zero-shot-object-detection-on-ms-coco → SeeDS (2023), ZSD-SCR (2022), ZSD-RRFS (2022)
        - https://paperswithcode.com/sota/zero-shot-object-detection-on-pascal-voc-07 → SeeDS (2023), ZSD-RRFS (2022)
        - https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0 → Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023)
        - https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0-val → Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023)
        - https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw → Grounding DINO 1.5 Pro (2024)
        - https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco → Grounding DINO 1.5 Pro (2024)

        best models: Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023), SeeDS (2023)

        best dataset: MS-COCO 







possible models:

- yolov9
- resnet
- imagenet
- COCA model



# 2. evaluate them against a variety of advx and captchas

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
