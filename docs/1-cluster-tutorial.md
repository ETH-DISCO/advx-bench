source: https://hackmd.io/hYACdY2aR1-F3nRdU8q5dA

meta:

- clusters are accessible to all lab students.
	- to use >8 gpus get your supervisor's permission first. check the calendar for high cluster usage times.
 	- A6000 and A100s require special privileges.
	- 
- clusters use a SLURM interface.
	- only submit to arton[01-08].

tutorials:

- https://gitlab.ethz.ch/disco-students/cluster (job scripts)
- https://computing.ee.ethz.ch/Programming/Languages/Conda (also see `netscratch` directory)
- https://computing.ee.ethz.ch/Services/SLURM
- https://computing.ee.ethz.ch/FAQ/JupyterNotebook?highlight=%28notebook%29 (jupyter notebook)

# setup

enter network:

- a) https://www.isg.inf.ethz.ch/Main/ServicesNetworkVPN (vpn)
- b) j2tik.ethz.ch (jumphost)

then ssh into the tik42 login node (shouldn't run any computation):

```bash
ssh ETH_USERNAME@tik42x.ethz.ch
``` 

add the following to your `~/.bashrc` file where `USER_PATH` should be the location of your conda installation, most likely under `/itet-stor/ETH_USERNAME/net_scratch`:

```bash
# conda installation
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
alias smon_free="grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt"
alias smon_mine="grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt"
alias watch_smon_free="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt\""
alias watch_smon_mine="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt\""
[[ -f USER_PATH/conda/bin/conda ]] && eval "$(USER_PATH/conda/bin/conda shell.bash hook)"

# current usage
alias smon_free="grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt"
alias smon_mine="grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt"`
```

also use [lilmamba / mamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) instead of conda because it's faster:

```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

store all your data in: `/itet-stor/ETH_USERNAME/net_scratch/YOUR_PROJECT` (not the same as `scratch_net`).

# checking resource availability

```bash
# checking available resources
smon_free
squeue --Format=jobarrayid:9,state:10,partition:14,reasonlist:16,username:10,tres-alloc:47,timeused:11,command:140,nodelist:20

# interactive session (permitted for prototyping)
srun  --mem=25GB --gres=gpu:01 --exclude=tikgpu[06-10] --pty bash -i

# jupyter notebook (assuming compute node already allocated)
# will host at something like `http://<hostname>.ee.ethz.ch:5998/?token=5586e5faa082d5fe606efad0a0033ad0d6dd898fe0f5c7af`
# port range [5900-5999]
conda create --name jupyternb notebook --channel conda-forge
conda activate jupyternb
jupyter notebook --no-browser --port 5998 --ip $(hostname -f)
```

check resource allocation with:

```python
import torch
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__CUDA Device Name:',torch.cuda.get_device_name(0))
print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
```

# batch job submission

you have to submit batch jobs to the slurm system.

- make sure the file is executable `chmod +x job.sh`
- you can also pass arguments `$1, $2, $3, ...` to a shell script by calling `sbatch job.sh arg1 arg2 arg3`.
- run `sbatch job.sh`
- update number of parallel jobs `scontrol update ArrayTaskThrottle=<count> JobId=<jobID>`

```bash
#!/bin/bash

# replace `DIRECTORY` with your own codebase, ie. /itet-store/ETH_USERNAME/net_scratch/projectX

#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/TODO_USERNAME/net_scratch/cluster/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/TODO_USERNAME/net_scratch/cluster/jobs/%j.err # where to store error messages
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu10,tikgpu[06-09]
#CommentSBATCH --nodelist=tikgpu01 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'


ETH_USERNAME=TODO_USERNAME
PROJECT_NAME=cluster
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=intro-cluster
mkdir -p ${DIRECTORY}/jobs
#TODO: change your ETH USERNAME and other stuff from above according + in the #SBATCH output and error the path needs to be double checked!

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# Execute your code
python main.py

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
```

keep your jobs short. set high "nice values" for your jobs so more important stuff has a priority. submit many small jobs in an array

```bash
sprio --long #shows queue, priority, and the priority factors
sbatch --nice=100000 job.sh #starts job with a high nice value
scontrol update job <JOB ID> nice=100000 #changes the nice value of a qued job (default and minimum value is 0)
```

<br><br>

# common errors

> "not enough space on device"

change the temporary directory which will be used by conda.

```bash
TMPDIR="/itet-stor/ETH_USERNAME/net_scratch/tmp/" && mkdir -p "${TMPDIR}" && export TMPDIR
```

> cuda version issues

set the following flag before your conda command.

```
CONDA_OVERRIDE_CUDA=11.7 conda ...
```

> what kind of nodes exist on the cluster?

2.2TB of GPU memory as of July 2023:

- GPU Nodes
	- 8x A100 with 80GB on tikgpu10
	- 8x A6000 with 48GB on tikgpu08
	- 24x RTX_3090 with 24GB on tikgpu[06,07,09]
	- 13x Titan RTX 24GB on tikgpu[04,05]
	- 21x Titan XP 12GB on tikgpu[01,02,03]
	- 2x Tesla V100 32GB on tikgpu05
	- 7x GeForce RTX2080 Ti 11GB on tikgpu01 and artongpu01
- CPU Nodes
	- 16x Dual Octa-Core Intel Xeon E5-2690 on each [arton01-03] with 125GB
	- 20x Dual Deca-Core Intel Xeon E5-2690 v2 on each [arton04-08] with 125GB
	- 20x Dual Deca-Core Intel Xeon E5-2690 v2 on each [arton09-10] with 251GB
	- 20x Dual Deca-Core Intel Xeon E5-2690 v2 on [arton11] with 535GB
