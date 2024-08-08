these are the best models as of august 2024.

pipeline:

- 1) generate images with flux.v1

    - https://huggingface.co/black-forest-labs/FLUX.1-dev (needs gpu)

- 2) caption images with llama3

    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5 (needs gpu)
    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4 (needs gpu)

- 3) classify with meta clip

    - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b (runs on laptop)

- 4) detect boundary boxes with grounding dino

    - https://huggingface.co/IDEA-Research/grounding-dino-base (runs on laptop)

- 5) segment images with sam vit 2 (using boundary boxes from previous step)

    - https://huggingface.co/facebook/sam2-hiera-small (needs gpu)


for each category we also have weaker models that can run on cpu/mps architectures. but the largest / top performing models are so huge that they can't run on colab's free tier. so you either have to pay up or use the gpu cluster.
