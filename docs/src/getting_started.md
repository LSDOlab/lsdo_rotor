# Getting started
This page provides instructions for installing CADDEE.

## Installation

Depending on the user's operating system, the installation instructions may be different.

## Windows

### Step 0: Installing WSL
CADDEE's geometry engine uses software that is incompatible with Windows at this time. As a simple workaround, the quickest way to install CADDEE on Windows is through [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL), which requires Windows 10 (version 2004 or higher) or Windows 11. The default Linux distribution installed by WSL is Ubuntu.
Once WSL is sucessfully installed, open an (Ubuntu) terminal window (simply type Ubuntu in the Windows search bar) and execute the following commands. 

### Step 1: Update packages
```sh
$ sudo apt-get update
$ sudo apt-get dist-upgrade
$ sudo apt-get install build-essential
$ sudo apt-get install ffmpeg libsm6 libxext6
```

### Step 2: Install miniconda
Next, we recommend the usage of Python environments. [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) is a compact version of the Anaconda Python distribution. Please see the link to find detailed installation instructions and verify that the commands below to install Miniconda are up-to-date. 

```sh
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm -rf ~/miniconda3/miniconda.sh
```

### Step 3: Create conda environment
Now, create a Conda environment. We recommend Python 3.9. Afterward, activate the environment.

```sh
$ conda create --name caddee python==3.9
$ conda activate caddee
```

Note that in the command above `caddee` is the name of the environment and be anything. 

### Step 4: Install CADDEE and other packages
Next, we recommend creating an easily accessable directory on your local machine where you install CADDEE and any other packages that may be necessary. The command  `mkdir packages` will make a sub-directory called 'packages' inside your current direcory and you can change your directory accordingly by typing `cd packages`. If you already have central directory where you would like to install all your packages in, change the directory by typing `cd path/to/desired/directory`. Once you are in the right directory execute the following commands. 

```sh
$ git clone https://github.com/LSDOlab/caddee_new.git
$ cd caddee
$ pip install -e .
$ cd ..
```

Note that this installs CADDEE into the specified Conda enveironment and everytime you open a new Ubuntu terminal window, you need to activate the conda environment. Future versions of CADDEE can be 'pulled' from github by typing

```sh
$ git pull
```

While most packages needed to run CADDEE will be installed automatically, there are a number of packages that you may need for certain (physics-based) simulations. We recommend the following packages to be installed additionally:

#### Geometry engine lsdo_geo: 
This package is needed for any analysis that involves a central geometry or changes to it.

```sh
$ git clone https://github.com/LSDOlab/lsdo_geo.git
$ cd lsdo_geo
$ pip install -e .
$ cd ..
```

#### Vortex-based Aerodynamic Solver Toolkit (VAST): 
This package is needed to perform pyhsics-based aerodynamic analysis using vortex theory via the steady or unsteady vortex-lattice methd (VLM)

```sh
$ cd ..
$ git clone https://github.com/jiy352/VAST.git
$ cd VAST
$ pip install -e .
$ cd ..
```

#### lsdo_rotor:
This package is needed to perform rotor aerodynamic analysis through blade element momentum (BEM) theory and steady-state dynamic inflow solver (Pitt--Peters).

```sh
$ git clone https://github.com/LSDOlab/lsdo_rotor.git
$ cd lsdo_rotor
$ pip install -e .
$ cd ..
```

#### Aframe:
This package is needed for 1-D beam structural analysis

```sh
$ git clone https://github.com/LSDOlab/aframe.git
$ cd aframe
$ pip install -e .
$ cd ..
```

## Ubuntu
The installation instructions for Ubuntu are the same from <u>step 1 onward</u>.

## MacOS
The installation instructions for MacOS are the same from <u> step 2 onward</u>. It is important to note that the commands for installing miniconda are slightly different to taken into account the correct operating system.

```sh
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
