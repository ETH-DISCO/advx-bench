see: https://github.com/ETH-DISCO/cluster-tutorial/blob/main/README.md#2-using-apptainer

```bash
# install deps
pip install ipywidgets jupyterlab_widgets --no-cache-dir --user --verbose
pip install torch torchvision torchaudio --no-cache-dir --user --verbose
pip install huggingface_hub pynvml accelerate numpy diffuser transformers --no-cache-dir --user --verbose

# update path after installing
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# huggingface login for gated models
huggingface-cli login
```

selected models:

- step 1) generate images with flux.v1

    - https://huggingface.co/black-forest-labs/FLUX.1-dev (needs gpu)

- step 2) caption images with llama3

    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5 (needs gpu)
    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4 (needs gpu)

- step 3) classify with meta clip

    - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b (runs on laptop)

- step 4) detect boundary boxes with grounding dino

    - https://huggingface.co/IDEA-Research/grounding-dino-base (runs on laptop)

- step 5) segment images with sam vit 2 (using boundary boxes from previous step)

    - https://huggingface.co/facebook/sam2-hiera-small (needs gpu)
