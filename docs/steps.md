conference: https://advml-frontier.github.io/

turlan's repo: https://gitlab.ethz.ch/disco-students/fs24/image-captchas

next steps:

- start writing, try to present the results we've already got

    - start writing first, don't expand the project in too far
    - ie. "all of our models perform badly on the masked images" -> generate datasets, plot metrics such as acc@1, acc@5, write ~6 pages (see: https://pytorch.org/vision/stable/datasets.html)
    - mask variables (ie. alpha intensity, ...) -> 4 patterns are enough for now
    - word clouds
    - resolution (try higher res datasets with >256px, maybe downscale if too large), take a dozen images per category, a fraction will suffice
    - use standard cls/seg datasets

- use the labels from the models train set (closed vocabulary)
- use cosine distance in latent space to measure cls error
- use chatgpt key for now instead of llamav3

rough paper structure:

- image models are struggeling -> give some motivating examples, ie. the sam2 error
- related work
- show which examples the models have been struggling on (turlans results) and what we can learn from that
- try natural adversarial examples and out of distribution datasets (find image models that have been trained on this)
- apply some of these masks / filters, then adversarially train and measure robustness
