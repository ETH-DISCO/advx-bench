conference: https://advml-frontier.github.io/

turlan's repo: https://gitlab.ethz.ch/disco-students/fs24/image-captchas

idea: adversarial images cannot be used for captchas.

steps:

1. do quantitative evaluation of hcaptcha images we scraped so far

    - label a bunch of hcaptcha images manually
    - evaluate some cls/det/seg models on these (not visual reasoning)

2. generate some synthetic hcaptcha images
    
    - learn from insights from step 1, check wha tthe models failed on

    - recreate hcaptcha images (instead of labeling them) -> study noise, distortions, perlin noise patches, circled gradients, etc.
    - fine-tune / robustify a model on these -> or use existing models like ASAM (https://arxiv.org/abs/2405.00256)
    - evaluate the model on hcaptcha images

3. present findings

    - see at what point it's too hard for humans vs. models
    - check if robustified model has lower accuracy

solver framework: https://github.com/QIN2DIM/hcaptcha-challenger


# step 1

running models:

- step 1) caption images with llama3

    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5 (needs gpu)
    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4 (needs gpu, int4 quantized)

- step 2) classify with meta clip

    - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b ✅ (runs on laptop)

- step 3) detect boundary boxes with grounding dino

    - https://huggingface.co/IDEA-Research/grounding-dino-base ✅ (runs on laptop)

- step 4) segment images with sam vit 2 (using boundary boxes from previous step)

    - https://huggingface.co/facebook/sam2-hiera-small (needs gpu)





<!--

# step 2

generating synthetic hcaptcha images:

- https://huggingface.co/black-forest-labs/FLUX.1-dev (needs gpu)

-->

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
