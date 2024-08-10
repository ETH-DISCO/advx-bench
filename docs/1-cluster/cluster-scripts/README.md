# Hitchhiker's Guide To The Cluster

## Setup Conda

Next you should setup up your conda according to the Hitchhikers guide.
Make sure you also append the necessary stuff to your bashrc file, including the smon_free aliases.

## Get an interactive session

Once you are in tik42x you can execute the following to get an interactive bash session using one gpu on tikgpu03

```
srun  --mem=25GB --gres=gpu:01 --nodelist tikgpu03 --pty bash -i
```

you can check the current job queue using

```
squeue --Format=jobarrayid:9,state:10,partition:14,reasonlist:16,username:10,tres-alloc:47,timeused:11,command:140,nodelist:20 
```

tipp: by adding a `| grep USERNAME` at the end you can filter the displayed list

This repository contains a sample MNIST script in `main.py`. 
First, create the conda environment using
`conda env create -f env.yml`
then you can activate it using `conda activate intro-cluster`.
Afterwards you can simply call `python main.py` and your MNIST training should start, double check the output on what type of GPU are you running?

## Submit your first job

We only use interactive sessions for debugging or prototyping and submit the rest of the jobs using jobscripts.
A simple sample jobscript is provided in `job.sh`. NOTE: You have to adjust some parameters inside the script such as your username and the right directories!

Then you can submit it using `sbatch job.sh`.

## Submit your first array job

Similarly if you have lots of jobs, you can use an array job to start them all and make sure that only x of them are running at the same time.
A similar sample script is proivded and you can use `sbatch array_job.sh` to try it.
