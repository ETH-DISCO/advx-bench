conference: https://advml-frontier.github.io/

idea: adversarial images cannot be used for captchas.

steps:

1. do quantitative evaluation of hcaptcha images we scraped so far

    - label a bunch of hcaptcha images manually
    - evaluate some cls/det/seg models on these (not visual reasoning)

2. generate some synthetic hcaptcha images

    - recreate hcaptcha images (instead of labeling them) -> study noise, distortions, perlin noise patches, circled gradients, etc.
    - fine-tune / robustify a model on these -> or use existing models like ASAM (https://arxiv.org/abs/2405.00256)
    - evaluate the model on hcaptcha images

3. present findings

    - see at what point it's too hard for humans vs. models
    - check if robustified model has lower accuracy



# step 1

- [x] scrape images from hcaptcha -> dune by turlan: https://gitlab.ethz.ch/disco-students/fs24/image-captchas
- [x] get ground truth -> done by turlan in [labelstudio](https://labelstud.io/): https://gitlab.ethz.ch/disco-students/fs24/image-captchas/-/blob/main/assets/datasets/hcaptcha_dataset_turlan/project-2-at-2024-07-08-17-39-4b3e31b4.zip
- [ ] run solvers via gpu cluster while turlan is labeling




<!--

adversarial examples:

- torchattack library
- roz:
    - https://github.com/wang-research-lab/roz/blob/main/scripts/common_adversarial_attack/attack.py
    - https://github.com/wang-research-lab/roz/blob/main/download_cifar.py
    - https://github.com/wang-research-lab/roz/blob/main/scripts/common_adversarial_attack/run_common_adversarial_attack.py
- pwc:
    - https://paperswithcode.com/task/adversarial-attack
    - https://paperswithcode.com/task/real-world-adversarial-attack
    - https://paperswithcode.com/task/adversarial-attack-detection

naturally occurring adversarial examples:

- https://github.com/hendrycks/natural-adv-examples

datasets:

- aggregators:
    - https://datasetninja.com/
    - https://huggingface.co/datasets
- captchas:
    - https://github.com/orlov-ai/hcaptcha-dataset
    - https://github.com/Inefficacy/Captcha-Datasets
    - https://www.kaggle.com/datasets/mikhailma/test-dataset
    - https://www.kaggle.com/datasets/cry2003/google-recaptcha-v2-images
    - https://datasetninja.com/google-recaptcha-image

-->
