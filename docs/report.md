we want to make robust vision based turing tests using advx.

# 1. find the strongest models

leaderboards:

- https://paperswithcode.com/area/computer-vision ⭐️ ← most up to date, most comprehensive
- https://segmentmeifyoucan.com/leaderboard
- https://huggingface.co/spaces/hf-vision/object_detection_leaderboard 
- https://cocodataset.org/#detection-leaderboard (outdated, from 2020)

tasks:

- classification: image class labels
    
    - https://paperswithcode.com/area/computer-vision/classification-1
    - https://paperswithcode.com/area/computer-vision/image-classification
    - https://paperswithcode.com/task/zero-shot-transfer-image-classification ⭐️
        - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-5

- segmentation: pixel groups (sometimes with labels)

    - https://paperswithcode.com/area/computer-vision/semantic-segmentation
    - https://paperswithcode.com/area/computer-vision/2d-semantic-segmentation

- object detection: bounding boxes with labels
    
    - https://paperswithcode.com/area/computer-vision/object-detection
    - https://paperswithcode.com/area/computer-vision/2d-object-detection








possible models:

- yolov9
- ViT (can be loaded directly into pytorch)
- resnet (can be loaded directly into pytorch)
- imagenet
- consider fine tuning
- COCA model (works best on task 2)

datasets:

- https://datasetninja.com/
- https://huggingface.co/datasets

- https://github.com/orlov-ai/hcaptcha-dataset
- https://github.com/Inefficacy/Captcha-Datasets
- https://www.kaggle.com/datasets/mikhailma/test-dataset
- https://www.kaggle.com/datasets/cry2003/google-recaptcha-v2-images
- https://datasetninja.com/google-recaptcha-image



# 2. evaluate them against a variety of advx and captchas

generating advx:

- https://paperswithcode.com/task/adversarial-attack
- https://paperswithcode.com/task/real-world-adversarial-attack
- https://paperswithcode.com/task/adversarial-attack-detection
- see: http://videos.rennes.inria.fr/seminaire-SoSySec/Maura-Pintor-03-02-2023/20230203-Maura-Pintor-sosysec-slides.pdf
- see: `torchattack`

datasets:

- https://github.com/hendrycks/natural-adv-examples
- try rotating images
- try selecting a subset of images ("only select 2 horses of 4")
- try to integrate things that are invisible to humans but visible to models (ie. low transparency, undersaturated values)
- try out new hcaptcha dataset with ascii characters and funny colors (reach out to turlan)
    - https://gitlab.ethz.ch/disco-students/fs24/image-captchas
    - https://gitlab.ethz.ch/disco-students/fs24/image-captchas/-/blob/main/assets/datasets/hcaptcha_dataset_turlan/processed/0a28e4fc9452e9b90e2e08e564de91754b85db04dba33f7216378aac883f5d3f.png?ref_type=heads
