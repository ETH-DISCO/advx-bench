setup

```bash
# login
ssh yjabary@tik42x.ethz.ch

# convenience commands for slurm
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
alias smon_free="grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt"
alias smon_mine="grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt"
alias watch_smon_free="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt\""
alias watch_smon_mine="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt\""

# troubleshooting for common issues
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=UTF-8
export LANG=C.UTF-8
```

using a notebook

```bash
# attack to available compute node
smon_free
srun  --mem=25GB --gres=gpu:01 --nodelist tikgpu06 --pty bash -i

# set up storage
mkdir -p /scratch/yjabary
cd /scratch/yjabary

# set up jupyter notebook
conda create --name jupyternb notebook --channel conda-forge
conda activate jupyternb
jupyter notebook --no-browser --port 5998 --ip $(hostname -f) # port range [5900-5999]

# exit
conda deactivate
conda env list
conda remove --all --yes --name jupyternb
exit
```

```bash
pip install <dependency> --upgrade --no-cache-dir --user --verbose
```

---

these are the best models as of august 2024.

pipeline:

- step 1) generate images with flux.v1

    - https://huggingface.co/black-forest-labs/FLUX.1-dev (needs gpu)
    - https://fal.ai/models/fal-ai/flux-realism

- step 2) caption images with llama3

    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5 (needs gpu)
    - https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4 (needs gpu)

- step 3) classify with meta clip

    - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b (runs on laptop)

- step 4) detect boundary boxes with grounding dino

    - https://huggingface.co/IDEA-Research/grounding-dino-base (runs on laptop)

- step 5) segment images with sam vit 2 (using boundary boxes from previous step)

    - https://huggingface.co/facebook/sam2-hiera-small (needs gpu)

for each category we also have weaker models that can run on cpu/mps architectures. but the largest / top performing models are so huge that they break google colab's free tier. so you either have to pay up or use the gpu cluster.
