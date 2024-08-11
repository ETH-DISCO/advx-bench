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
