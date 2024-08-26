conference: https://advml-frontier.github.io/

turlan's repo: https://gitlab.ethz.ch/disco-students/fs24/image-captchas

---

1. how well can we solve hcaptcha? where do our models suck? (turlans eval + yahyas eval)
2. how well do the masks and distortions they're using generalize?
3. can we robustify our models against them?

---

next steps:

- start writing, try to present the results we've already got

    - start writing first, don't expand the project in too far
    - ie. "all of our models perform badly on the masked images" -> generate datasets, plot metrics such as acc@1, acc@5, write ~6 pages (see: https://pytorch.org/vision/stable/datasets.html)
    - mask variables (ie. alpha intensity, ...) -> 4 patterns are enough for now
    - word clouds
    - resolution (try higher res datasets with >256px, maybe downscale if too large), take a dozen images per category, a fraction will suffice
    - use standard cls/seg datasets
