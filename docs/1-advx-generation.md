try adversarial attacks on the chosen models

see: https://github.com/wang-research-lab/roz/blob/main/scripts/common_adversarial_attack/attack.py

see:

- https://github.com/wang-research-lab/roz/blob/main/download_cifar.py
- https://github.com/wang-research-lab/roz/blob/main/scripts/common_adversarial_attack/run_common_adversarial_attack.py

generating advx:

- https://paperswithcode.com/task/adversarial-attack
- https://paperswithcode.com/task/real-world-adversarial-attack
- https://paperswithcode.com/task/adversarial-attack-detection
- see: http://videos.rennes.inria.fr/seminaire-SoSySec/Maura-Pintor-03-02-2023/20230203-Maura-Pintor-sosysec-slides.pdf
- see: `torchattack`

*datasets:*

- data generation using flux
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

registry:

- https://github.com/wang-research-lab/roz/blob/6b4b7ff9d98a0a6fb4aeb4512859c1a0b16a0138/scripts/natural_distribution_shift/src/registry.py
- https://github.com/wang-research-lab/roz/blob/6b4b7ff9d98a0a6fb4aeb4512859c1a0b16a0138/scripts/natural_distribution_shift/src/inference.py#L115
