# Getting started
This page provides instructions for installing LSDO Rotor.

## Installation



### Step 0: Install miniconda
Next, we recommend the usage of Python environments. [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) is a compact version of the Anaconda Python distribution. Please see the link to find detailed installation instructions and verify that the commands below to install Miniconda are up-to-date. 

```sh
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm -rf ~/miniconda3/miniconda.sh
```

### Step 1: Create conda environment
Now, create a Conda environment. We recommend Python 3.9. Afterward, activate the environment.

```sh
$ conda create --name lsdo_rotor python==3.9
$ conda activate lsdo_rotor
```

Note that in the command above `lsdo_rotor` is the name of the environment and be anything. 

### Step 2: Install LSDO Rotor
Next, we recommend creating an easily accessable directory on your local machine where you install LSDO rotor and any other packages that may be necessary. The command  `mkdir packages` will make a sub-directory called 'packages' inside your current direcory and you can change your directory accordingly by typing `cd packages`. If you already have central directory where you would like to install all your packages in, change the directory by typing `cd path/to/desired/directory`. Once you are in the right directory execute the following commands. 

```sh
$ git clone https://github.com/LSDOlab/lsdo_rotor.git
$ cd lsdo_rotor
$ pip install -e .
$ cd ..
```

Note that this installs lsdo_rotor into the specified Conda enveironment and everytime you open a new Ubuntu terminal window, you need to activate the conda environment. Future versions of lsdo_rotor can be 'pulled' from github by navigating into the local (your machine) directory containing lsdo_rotor and typing

```sh
$ git pull
```

### Step 3: Running an example script
To see whether the installation was successful, you can try running an example script by typing the following commands

```sh
$ cd examples
$ python ex_BEM_rotor_optimization.py
```