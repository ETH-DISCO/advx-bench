> create a benchmarking pipeline. don't use the captcha clone. you're on the defense / captcha-generation side, so you own the models anyway.

# 1. find strong detection + segmentation models

models:

- https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-5

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

# 2. check how well advx works on these models

generating advx:

- see: http://videos.rennes.inria.fr/seminaire-SoSySec/Maura-Pintor-03-02-2023/20230203-Maura-Pintor-sosysec-slides.pdf
- see: `torchattack`

datasets:

- https://github.com/hendrycks/natural-adv-examples
- try rotating images
- try selecting a subset of images ("only select 2 horses of 4")
- try to integrate things that are invisible to humans but visible to models (ie. low transparency, undersaturated values)
- try out new hcaptcha dataset with ascii characters and funny colors (reach out to turlan)
    - check out the adversarial images Turlan has collected from hCaptcha: https://gitlab.ethz.ch/disco-students/fs24/image-captchas
    - https://gitlab.ethz.ch/disco-students/fs24/image-captchas/-/blob/main/assets/datasets/hcaptcha_dataset_turlan/processed/0a28e4fc9452e9b90e2e08e564de91754b85db04dba33f7216378aac883f5d3f.png?ref_type=heads
