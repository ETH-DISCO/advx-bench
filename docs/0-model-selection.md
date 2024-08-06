we want to improve vision-based turing-tests using advx.

we're looking for the strongest zero shot image recognition models (umbrella term) as of august 2024.

these are our findings.

# 1. performance benchmarks

overview:

- huggingface: https://huggingface.co/docs/transformers/en/model_doc/beit
- timm: https://huggingface.co/docs/timm/en/models
- course: https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/pre-intro

## 1.1. RoZ benchmark

> see: https://arxiv.org/pdf/2403.10499 (CVPR 2024)
>
> has the same goal as us - but only focuses on CLIP models for cls tasks on ImageNet
>
> important learnings:
> 
> - CLIP isn't as adversarially robust as advertised
> - ViT is more adversarially robust than RN

ranking by performance (based on table 2)

- ViT-B/16 (Standard ImageNet model) - 84.20% accuracy
- ResNet50x16 (CLIP model) - 70.67% accuracy
- ViT-B/32 (Standard ImageNet model)
- ResNet50x4 (CLIP model)
- ResNet101 (Standard ImageNet model)

ranking by adversarial robustness (based on table 1, figure 8)

- ViT-B/16 (Standard ImageNet model)
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

detection:

- https://paperswithcode.com/area/computer-vision/object-detection
- https://paperswithcode.com/area/computer-vision/2d-object-detection

segmentation:

- https://paperswithcode.com/area/computer-vision/semantic-segmentation (overview)
- https://paperswithcode.com/area/computer-vision/2d-semantic-segmentation (overview)
- https://paperswithcode.com/task/image-segmentation
- https://paperswithcode.com/task/semantic-segmentation
- https://paperswithcode.com/task/universal-segmentation (no data)
- https://paperswithcode.com/task/zero-shot-semantic-segmentation
- https://paperswithcode.com/task/zero-shot-segmentation (zero shot segmentation is actually useless)
    - https://paperswithcode.com/sota/zero-shot-segmentation-on-segmentation-in-the → Grounded HQ-SAM (2023), Grounded-SAM (2023)
    - https://paperswithcode.com/sota/zero-shot-segmentation-on-ade20k-training → GEM MetaCLIP (2023) (← this doesn't make sense, it's a classification model)

zero shot classification:

- https://paperswithcode.com/task/zero-shot-transfer-image-classification
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-6 → LiT-22b (2023), LiT ViT-e (2022), CoCa (2022), EVA-CLIP-18B (2024)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-4 → CoCa (2022), LiT ViT-e (2022), LiT-22B (2023), EVA-CLIP-18B (2024)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-5 → CoCa (2022), LiT-22B (2023), LiT ViT-e (2022), EVA-CLIP-18B (2024)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-3 → LiT-22B (2023), CoCa (2022), LiT ViT-e (2022), LiT-tuning (2021)
    - https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-1 → M2-Encoder (2024), CoCa (2022), LiT-22B (2023), LiT ViT-e (2022)

zero shot object detection:

- https://paperswithcode.com/task/zero-shot-object-detection
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-ms-coco → SeeDS (2023), ZSD-SCR (2022), ZSD-RRFS (2022)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-pascal-voc-07 → SeeDS (2023), ZSD-RRFS (2022)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0 → Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0-val → Grounding DINO 1.5 Pro (2024), OWLv2 (2023), MQ-GLIP-L (2023)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw → Grounding DINO 1.5 Pro (2024)
    - https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco → Grounding DINO 1.5 Pro (2024)

zero shot semantic segmentation:

- https://paperswithcode.com/task/zero-shot-semantic-segmentation
    - https://paperswithcode.com/sota/zero-shot-semantic-segmentation-on-coco-stuff → OTSeg+ (2024), JSeg (2024), OTSeg (2024), ZegCLIP (2022)
    - https://paperswithcode.com/sota/zero-shot-semantic-segmentation-on-pascal-voc → OTSeg+ (2024), OTSeg (2024), CLIP-RC (2024), ZegCLIP (2022)
    - https://paperswithcode.com/sota/zero-shot-semantic-segmentation-on-mess → CAT-SEG-L (2024)
    - https://paperswithcode.com/sota/zero-shot-semantic-segmentation-on-ade20k-847 → MAFT (2023)

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

# 2. chaining models

it's possible to chain models by ie. generating a prompt with one model and then using that prompt with another model etc.

*image to text / image captioning*

- using image captions to generate textual prompts for CV models
- the models must be open vocabulary to be useful, otherwise they're constrained to the labels they were trained on (just like detr)
- https://paperswithcode.com/task/image-captioning -> BLIP-2 ViT-G FlanT5 XL (not open vocab)
- https://huggingface.co/tasks/image-to-text
- https://huggingface.co/models?pipeline_tag=image-to-text
- https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/tasks-models-part1#image-captioning
- examples:
    - blip 1:
        - https://huggingface.co/Salesforce/blip-vqa-base
    - blip 2:
        - https://huggingface.co/Salesforce/blip2-opt-2.7b
        - https://huggingface.co/Salesforce/blip2-opt-2.7b-coco
        - https://huggingface.co/Salesforce/blip2-flan-t5-xxl
        - https://huggingface.co/Salesforce/instructblip-vicuna-7b (also custom instructions)
    - git:
        - https://huggingface.co/microsoft/git-large-coco
        - https://huggingface.co/microsoft/git-base
        - https://huggingface.co/microsoft/git-base-vqav2
    - pix2struct:
        - https://huggingface.co/google/pix2struct-textcaps-base
    - gpt2:
        - https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

*text to image / image generation*

- using the same caption to generate an image to then use that image as a prompt for another model
- https://huggingface.co/tasks/text-to-image
- https://huggingface.co/models?pipeline_tag=image-to-text
- https://huggingface.co/black-forest-labs/flux (state of the art)

# 3. ease of use

## 3.1. cls

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
    - https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    - https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb
    - https://huggingface.co/openai/clip-vit-large-patch14
    - also roz: https://github.com/wang-research-lab/roz

open-coca @ 2022:

- https://arxiv.org/abs/2212.07143
- coca: https://arxiv.org/abs/2205.01917, https://github.com/lucidrains/CoCa-pytorch
- ✅ installation:
    - https://github.com/mlfoundations/open_clip

eva / eva-clip-18b @ 2023:

- https://arxiv.org/pdf/2402.04252
- https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B
- clip with 18 billion parameters
- trained using apex and xformer
- ✅ installation:
    - models provided with paper are way too large (35.3 GB) and slow to be a feasible option
        - https://huggingface.co/papers/2402.04252
        - https://huggingface.co/models?search=evaclip
        - https://huggingface.co/BAAI/EVA-CLIP-8B (smallest model)
        - https://huggingface.co/BAAI/EVA-CLIP-18B
        - https://huggingface.co/BAAI/EVA-CLIP-8B-448
    - use open_clip's base model instead (just 2GB and reasonable inference times)

gem / metaclip @ 2024:

- https://arxiv.org/pdf/2309.16671
- https://github.com/facebookresearch/MetaCLIP
- beats clip and openclip by a wide margin in zero-shot classification
- massive speedup possible through flash attention and scaled dot product attention (but requires a gpu): https://huggingface.co/docs/transformers/main/en/model_doc/clip#expected-speedups-with-flash-attention-and-sdpa
- ✅ installation:
    - https://github.com/facebookresearch/MetaCLIP?tab=readme-ov-file#quick-start
    - https://huggingface.co/models?search=metaclip
    - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b (largest version)

google efficientnet @ 2020:

- https://arxiv.org/pdf/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- not on any leaderboards
- ✅ installation:
    - https://huggingface.co/google/efficientnet-b0
    - https://huggingface.co/docs/timm/en/models/efficientnet

google lit @ 2021:

- https://arxiv.org/abs/2111.07991
- https://github.com/google-research/vision_transformer
- https://github.com/google-research/vision_transformer/blob/main/model_cards/lit.md#model-data
- https://github.com/google-research/vision_transformer?tab=readme-ov-file#lit-models
- based on clip, but beats it in zero-shot classification
- ❌ installation:
    - https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb
    - somehow figured it out myself but it wasn't well documented and i'm not sure if what i did was correct

## 3.2. det

grounding dino @ 2024:

- https://arxiv.org/abs/2303.05499
- https://github.com/IDEA-Research/GroundingDINO
- dino is an improvement over detr: https://arxiv.org/abs/2005.12872 → https://arxiv.org/abs/2203.03605
- bleeding edge, still actively being developed
- ✅ installation:
    - https://huggingface.co/IDEA-Research/grounding-dino-base (largest version)
    - https://huggingface.co/IDEA-Research/grounding-dino-tiny

owlv2 / owlvit @ 2023:

- https://arxiv.org/abs/2305.01917
- ✅ installation:
    - https://huggingface.co/google/owlv2-base-patch16-ensemble → uses clip as backbone with vit-B/16
    - https://huggingface.co/google/owlvit-base-patch32 → uses clip as backbone with vit-B/32 (slower but more accurate)

detr @ 2022:

- https://arxiv.org/abs/2203.03605
- not useful for zero-shot semantic open vocabulary segmentation (although it does have some relevant features)
- ✅ installation:
    - extremely well documented, very easy to use
    - https://github.com/facebookresearch/detr

mq-det @ 2023:

- https://arxiv.org/pdf/2305.18980v2
- https://github.com/yifanxu74/mq-det
- ❌ installation:
    - no link to pretrained models, no documentation

grounding dino 1.5 @ 2024:

- https://arxiv.org/abs/2303.05499
- https://github.com/IDEA-Research/Grounding-DINO-1.5-API
- bleeding edge, still actively being developed
- ❌ installation:
    - only through paid api https://deepdataspace.com/request_api, not open source

## 3.3. semantic seg

> the majority of these models are not open vocabulary / constrained to the labels they were trained on + only accessible through `.pth` checkpoints that must be manually downloaded in addition to their repository and some other back bone (usually this process is not well documented)
> 
> this really narrows down the options

mask2former @ 2022:

- https://arxiv.org/abs/2112.01527
- https://github.com/facebookresearch/Mask2Former
- https://huggingface.co/models?other=mask2former
- https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md#coco-model-zoo
- not on leaderboard, not open vocabulary (limited to COCO or AD20k classes)
- ✅ installation:
    - see "resources" in the mask2former documentation
    - https://huggingface.co/docs/transformers/main/en/model_doc/mask2former
    - https://huggingface.co/blog/Mask2Former
    - https://huggingface.co/facebook/mask2former-swin-large-coco-instance (largest version)

clipseg @ 2021:

- https://arxiv.org/abs/2112.10003
- not on leaderboard but very popular
- ✅ installation:
    - https://huggingface.co/docs/transformers/main/en/model_doc/clipseg

grounding sam @ 2024:

- https://arxiv.org/abs/2303.05499
- https://paperswithcode.com/paper/grounding-dino-marrying-dino-with-grounded
- https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file
- https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino
- particularly relevant as SAM2 just came out
- not on leaderboard but very popular
- ✅ installation:
    - https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

otseg+ @ 2024:

- https://arxiv.org/pdf/2403.14183
- https://github.com/cubeyoung/OTSeg
- top perf
- installation:
    - https://github.com/cubeyoung/OTSeg?tab=readme-ov-file#pretrained-models

maft @ 2023:

- https://arxiv.org/abs/2310.00240
- https://github.com/jiaosiyu1999/MAFT
- top perf
- installation:
    - https://github.com/jiaosiyu1999/MAFT?tab=readme-ov-file#pretrained-weights

clip-dinoiser @ 2023:

- https://arxiv.org/abs/2312.12359
- https://github.com/wysoczanska/clip_dinoiser/
- adds DINO for open vocabulary semantic segmentation
- not on leaderboard - but has open vocabulary
- installation:
    - https://github.com/wysoczanska/clip_dinoiser/blob/main/demo.ipynb

zegformer @ 2022:

- https://arxiv.org/abs/2112.07910
- https://github.com/dingjiansw101/ZegFormer
- not on leaderboard - but has open vocabulary
- installation:
    - https://github.com/dingjiansw101/ZegFormer?tab=readme-ov-file#inference-demo-with-pre-trained-models

cliprc @ 2024:

- https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Exploring_Regional_Clues_in_CLIP_for_Zero-Shot_Semantic_Segmentation_CVPR_2024_paper.pdf
- https://github.com/uyzhang/CLIP-RC
- top perf
- installation:
    - https://github.com/uyzhang/CLIP-RC?tab=readme-ov-file#pretrained-models

zegclip @ 2022:

- https://arxiv.org/abs/2212.03588
- https://github.com/ZiqinZhou66/ZegCLIP
- top perf
- installation:
    - https://github.com/ZiqinZhou66/ZegCLIP/tree/main#pretrained-models

catseg @ 2024:

- https://arxiv.org/abs/2303.11797
- https://github.com/KU-CVLAB/CAT-Seg
- top perf
- ❌ installation:
    - https://github.com/KU-CVLAB/CAT-Seg?tab=readme-ov-file#pretrained-models
    - you must download weights + repository
    - doesn't have any documentation, just an evaluation script that only runs with nvidia gpus

detr @ 2022:

- same model as above
- not on leaderboard but very popular
- ❌ zero shot, doesn't need any textual prompts and finds all labels in the image - but only works with COCO classes
    - could maybe be fixed through ov-detr: https://github.com/yuhangzang/OV-DETR

## 3.3. semantic seg (with bounding boxes as queries)

sam vit @ 2023:

- https://arxiv.org/pdf/2304.02643
- https://github.com/facebookresearch/segment-anything
- ✅ installation:
    - very straightforward to use
    - https://huggingface.co/facebook/sam-vit-base
    - https://huggingface.co/facebook/sam-vit-huge (largest version)
    - https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
    - https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

sam vit 2 @ 2024 (published this week):

- https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/
- https://github.com/facebookresearch/segment-anything-2
- absolutely top perf
- ❌ installation:
    - only works with an nvidia gpu: `raise OSError('CUDA_HOME environment variable is not set. '`

grounded sam @ 2024:

- https://arxiv.org/abs/2401.14159
- https://github.com/IDEA-Research/Grounded-Segment-Anything
- https://huggingface.co/spaces/linfanluntan/Grounded-SAM
- ❌ installation:
    - docker container
    - docker doesn't support [apple mps](https://github.com/pytorch/pytorch/issues/81224) as a pytorch backend which then restricts the entire project to cpu or cuda backends

grounded hq-sam @ 2023:

- https://arxiv.org/abs/2306.01567
- https://github.com/SysCV/sam-hq
- ❌ installation:
    - model checkpoint must be manually downloaded from google drive link, no other way

## 3.4. image to text

open blip @ 2023:

- https://arxiv.org/abs/2312.03631
- finetuned on MS-COCO with the MOCHa RL framework
- https://assafbk.github.io/mocha/
- ✅ installation:
    - very straightforward
    - https://huggingface.co/moranyanuka/blip-image-captioning-large-mocha

blip 1 @ 2022:

- https://arxiv.org/abs/2201.12086
- not open vocabulary, constrained to coco, so no advantage over detr
- installation:
    - very straightforward to install, just 2 GB large
    - https://huggingface.co/Salesforce/blip-image-captioning-large

## 3.5. text to image

flux v1 @ 2024:

- no paper, just blog: https://blackforestlabs.ai/announcing-black-forest-labs/
- demo:
    - https://replicate.com/black-forest-labs/flux-schnell
    - https://fal.ai/models/fal-ai/flux/schnell
- ✅ installation:
    - really massive models, huge compute and memory requirements (can be over 50GB large even for the "schnell" version)
    - inference for smaller models is possible on cpu, stronger models can be accessed through an api
    - https://github.com/black-forest-labs/flux
    - https://huggingface.co/black-forest-labs/FLUX.1-dev
    - https://huggingface.co/black-forest-labs/FLUX.1-schnell
