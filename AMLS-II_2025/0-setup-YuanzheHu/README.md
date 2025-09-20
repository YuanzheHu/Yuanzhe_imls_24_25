[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xa-0YIVt)
![](figures/ucl-logo.svg)

# 0-SETUP
Welcome to ELEC0135 Advanced Machine Learning Systems II's labs! This repository contains instructions on how to setup everything you will need to solve them, as well as teaching you how to use conda python virtual environments to ensure reproducibility of your code.

> [!IMPORTANT]
> * Make sure to go through every items below before coming to the lab session, to minimise time spent troubleshooting IT issues and maximise time spent solving Machine Learning problems.


## Software and IDE Checklists

**OS**
> [!IMPORTANT]
> * Make sure your OS is updated to the latest version.

> [!CAUTION]
> * **Please switch the language of your OS to English, otherwise it will be very difficult for the teaching team to assist you in troubleshooting.**

**Git**
> [!IMPORTANT]
> * Install git on your computer from [the official git websiste](https://git-scm.com).

> [!TIP]
> * Make sure you're familiar with [git's basic commands](https://education.github.com/git-cheat-sheet-education.pdf)

**IDE for Python**

In this course, you will use both Python files and Jupiter Notebooks to implement various Machine Learning algorithms. You must install an 
Integrated Development Environment (IDE) that can execute both, such as Jupyter or VS Code. Although you are free to use the IDE of your choosing, we recommend you use VS Code, which you can install using the following instructions.

> [!IMPORTANT]
> * Install [VS Code](https://code.visualstudio.com)
> * In VS Code, go to the extensions tab, and download the [python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
> * If you are on a Windows laptop, please install the `Git bash terminal` by following [these instructions](https://www.geeksforgeeks.org/how-to-integrate-git-bash-with-visual-studio-code/).

**Cmake**
> [!IMPORTANT]
> * Install [Cmake](https://cmake.org/download/) as we will need it to compile the [dlib](http://dlib.net) librairy in lab 1.

## Virtual Environment Checklist

**Installing Conda**

Conda is an open-source package management and environment management tool that helps users install, run, and update software and dependencies. It works across multiple programming languages, including Python, R, and C/C++. Conda is widely used in data science, machine learning, and scientific computing because it simplifies the process of managing libraries and creating isolated environments, ensuring compatibility between dependencies.

> [!IMPORTANT]
> * Install [**Anaconda**](https://docs.anaconda.com/free/anaconda/) or an open-source equivalent (preferred), such as [**Miniconda**](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

**Setting up a virtual environment**

A virtual environment in Python is an isolated environment that allows you to manage and run a specific set of Python packages and dependencies independently from the system-wide Python installation. This is particularly useful for avoiding conflicts between packages when working on multiple projects with different requirements. Each virtual environment has its own Python interpreter and libraries, enabling you to install specific versions of packages for a project without affecting other projects or the global Python installation.

> [!IMPORTANT]
> - In the root directory (where you git cloned this lab and will clone all the others), create a `env` folder. 
> - In the `/env` folder, create an empty file with your notepad app and name it `environment.yml`.
> - Use the exemple to create your own environment file, with the environment name (not the file) `amls2` and the version of python 3.11.
> - Create a text file called `requirements.txt`, in which you will add the packages that will be needed for the different labs.
> - If you did everything right, you file structure should look like this, with `AMLS2_root` being the folder in which you git cloned the lab:
> 
> ```plaintext
> ├── AMLS2_root
> │   ├── env
> │   │   ├── environment.yml
> │   │   ├── requirements.txt
> │   ├── 0-SETUP
> │   │   ├── README.md
> │   │   ├── figures
> │   │   │   ├── ucl-logo.svg
> │   │   ├── examples
> │   │   │   ├── environment.yml
> │   │   │   ├── requirements.txt
> │   ├── 1-MULTI-LAYER-PERCEPTRON
>             .
>             .
>             .
> ```

> [!TIP]
> - We have included an example `environment.yml` file in this repository for you to understand the structure of a standard `environment.yml` file, where the pip requirements are stored separate into another file.
> - We have included an example `requirements.txt` file in this repository for you to understand the structure of a standard `requirements.txt` file, where the pip requirements are stored separate into another file.

> [!IMPORTANT]
> - Open a terminal and use cd to navigate to the root directory.
> - Run the command `sudo conda env create -f env/environment.yml`, enter the password of your session, press enter.
> - Run the command `conda info --envs` to check that the envirionment called daps has been created.

**Using the virtual environment**

> [!IMPORTANT]
> - In the terminal, you need to activate your virtual environment in order to execute programs within it. You can do so by running the command `conda activate amls2`.

> [!WARNING]
> - Sometimes, in VS Code's terminals, the package manager pip is not setup correctly and will not point to your virtual environment despite having run the activation command.
> - You can verify it by running the command `which pip`.
> - If `which pip` indicates another path than your virtual environment, then you should instead use a terminal outside of VS Code for pip related operations.

> [!IMPORTANT]
> - To import python packages to your virtual environment, add the package names (and version requirements if need be with [this syntax](https://iscompatible.readthedocs.io/en/latest/)) to the `env/requirements.txt` file, save it, then run:
> ```bash
> pip install -r requirements.txt
> ```

> [!IMPORTANT]
> - Make sure to run the Python Notebooks on the `amls2` kernel to use the packages you install.

