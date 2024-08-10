This guide will help you get started with using the TIK cluster at ETH Zurich.

# 1. SSH into the cluster

First enable your VPN connection to the ETH network.

- VPN documentation: https://www.isg.inf.ethz.ch/Main/ServicesNetworkVPN
- Based on my experience the openconnect CLI doesn't work. So i suggest downloading the the cisco anyconnect client and using the following settings:
	- server: `https://sslvpn.ethz.ch`
	- username: `<username>@student-net.ethz.ch`
	- password: your network password (also called Radius password, see: https://www.password.ethz.ch/)

Then ssh into the tik42 login node and use your default password (also called LDAPS/AD password).

```bash
ssh <username>@tik42x.ethz.ch
```

You can also configure a shortcut in your `~/.ssh/config` file to be able to ssh using just `ssh j2tik` or `ssh tik42x` in the future.

```
Host j2tik
  HostName j2tik.ethz.ch
  User <username>
Host tik42x
  HostName tik42x.ethz.ch
  User <username>
```

Once you're in you'll have access to:

- Compute: the login node is only for file management and job submission. do not run any computation on the login node. Run batch jobs on the compute nodes using the slurm system. The slurm system is a common job scheduler used in many HPC systems so it's a good skill to learn. You'll find plenty of resources online.
- Storage: use `/itet-stor/<username>/net_scratch` for your data.

Keep in mind:

- to use >8 GPUs you need your supervisor's permission and reserve the nodes in advance in the shared calendar.
- only submit jobs to `arton[01-08]`
	- see all available nodes using `sinfo`
	- the A100s with 80GB on `tikgpu10` need special privileges
	- the A6000s with 48GB on `tikgpu08` need special privileges
- set friendly `nice` values to your jobs and keep them short and efficient.

# 2. Setup

To set up your environment first install conda using the `conda_install.sh` script.

Optionally set up the "libmamba" solver for faster dependency resolution:

```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Next add the following instructions to your `~/.bashrc.<username>` file and then run `source ~/.bashrc.<username>`:

```bash
# convenience commands for slurm
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
alias smon_free="grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt"
alias smon_mine="grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt"
alias watch_smon_free="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt\""
alias watch_smon_mine="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt\""

# troubleshooting for common issues
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc
echo 'export LANG=en_US.UTF-8' >> ~/.bashrc
echo 'export LANGUAGE=en_US.UTF-8' >> ~/.bashrc
```

# 3. Submit batch jobs

First check which nodes are available:

```bash
smon_free

squeue --Format=jobarrayid:9,state:10,partition:14,reasonlist:16,username:10,tres-alloc:47,timeused:11,command:140,nodelist:20
```











This repository contains a sample MNIST script in `main.py`. 

First, create the conda environment using `conda env create -f env.yml` then you can activate it using `conda activate intro-cluster`.

Afterwards you can simply call `python main.py` and your MNIST training should start, double check the output on what type of GPU are you running?




## Submit your first job

We only use interactive sessions for debugging or prototyping and submit the rest of the jobs using jobscripts.
A simple sample jobscript is provided in `job.sh`. NOTE: You have to adjust some parameters inside the script such as your username and the right directories!

Then you can submit it using `sbatch job.sh`.



## Submit your first array job

Similarly if you have lots of jobs, you can use an array job to start them all and make sure that only x of them are running at the same time.
A similar sample script is proivded and you can use `sbatch array_job.sh` to try it.

















# Debugging

Job scripts don't show the output in real time.

For debugging purposes it might make sense to attach your terminal to individual compute nodes. This can be done using the following command (ie. for tikgpu06):

```bash
srun  --mem=25GB --gres=gpu:01 --nodelist tikgpu06 --pty bash -i
```

Next you can activate run a jupyter notebook on the compute node and access it through your browser using the following commands:

```bash
conda create --name jupyternb notebook --channel conda-forge
conda activate jupyternb
jupyter notebook --no-browser --port 5998 --ip $(hostname -f) # port range [5900-5999]
```

The hosted notebook will be available at a link similar to `http://<hostname>.ee.ethz.ch:5998/?token=5586e5faa082d5fe606efad0a0033ad0d6dd898fe0f5c7af` where the hostname is the same as the one you SSH'd into.

Make sure to deactivate the conda environment after you're done debugging:

```bash
conda remove --yes --name jupyternb --all
conda env list
conda deactivate
```

# References

Internal resources:

- outdated tutorial: https://hackmd.io/hYACdY2aR1-F3nRdU8q5dA
- up-to-date tutorial: https://gitlab.ethz.ch/disco-students/cluster
- conda install: https://computing.ee.ethz.ch/Programming/Languages/Conda
- slurm docs: https://computing.ee.ethz.ch/Services/SLURM
- jupyter notebook docs: https://computing.ee.ethz.ch/FAQ/JupyterNotebook?highlight=%28notebook%29
