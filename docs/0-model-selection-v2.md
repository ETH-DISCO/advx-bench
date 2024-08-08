pipeline:

- generate images with flux.v1

    - https://huggingface.co/black-forest-labs/FLUX.1-dev (needs gpu)
    - https://huggingface.co/black-forest-labs/FLUX.1-schnell (needs gpu)

- caption images with llama3

    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5 (needs gpu)
    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4 (needs gpu)

- classify images with meta clip

    - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b (runs on laptop)

- detect objects in images with grounding dino

    - https://huggingface.co/IDEA-Research/grounding-dino-base (runs on laptop)

- segment images with sam vit 2 using bounding boxes from grounding dino

    - https://huggingface.co/facebook/sam2-hiera-small (needs gpu)
