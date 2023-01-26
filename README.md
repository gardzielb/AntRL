# AntRL

## Installation

The following command can be used to install dependencies:
```shell
pip install -r requirements.txt
```

> **Attention!**
> Because of the compatibility issues with the libraries `stable-baselines3` and `sb3_contrib`, working on the project 
> required some changes to be made in their source code.

## Running the program

Ro tun the program the following command should be used:
```shell
python -m antrl -a ['ars' or 'sac'] <command> [options...]
```

For instance, to perform simple model evaluation with rendering, run
```shell
python -m antrl -a ars eval -r models/ARS/ars_d40_t20_h1_5M.zip
```
