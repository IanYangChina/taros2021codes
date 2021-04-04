#### Scripts for our Taros 2021 paper titled ''Open-sourcing, Enhancing the Multi-goal Reinforcement Learning Environment for Complex Robotics Manipulation using Pybullet''

This repo contains python and bash scripts to reproduce the experiment results proposed in our Taros 2021
paper, titled **Open-sourcing, Enhancing the Multi-goal Reinforcement Learning Environment for Complex Robotics 
Manipulation using Pybullet**.

#### Before running the scripts

1. Install the [`drl_implementation` package](https://github.com/IanYangChina/DRL_Implementation).
2. Install the [`pybullet_multigoal_gym` package](https://github.com/IanYangChina/pybullet_multigoal_gym).
3. Clone this repository to wherever you like.
4. On a terminal: `export PYTHONPATH=$PYTHONPATH:$PATH_OF_THE_PROJECT_ROOT`. 
Replace `$PATH_OF_THE_PROJECT_ROOT` with something like `/home/someone/xtyang_taros2021codes`.

#### Run the scripts

1. Single-step task:
from the project root, run `python run_single_step_task --task reach --hindsight`.

| | |
| :---------------------- | :----------------------------------------------- |
| Arguments               | Description                                      |
| `--task str`            | Task name, pick one in `['reach', 'push', 'pick_and_place', 'slide']` |
| `--num-runs`            | Number of task runs, each of which runs with a unique random seed |
| `--render`              | Use this flag if you want to render the task     |
| `--joint-ctrl`          | Use this flag to use joint control mode |
| `--hindsight`           | Use this flag to use hindsight experience replay with the 'future' strategy |

2. Multi-step task:
from the project root, run `python run_multi_step_task --task block_rearrange --num-blocks 3 --crcl`.

| | |
| :---------------------- | :----------------------------------------------- |
| Arguments               | Description                                      |
| `--task str`            | Task name, pick one in `['block_rearrange', 'block_stack', 'chest_pick_and_place', 'chest_push']` |
| `--num-runs`            | Number of task runs, each of which runs with a unique random seed |
| `--num-blocks`          | Number of blocks in a task |
| `--render`              | Use this flag if you want to render the task     |
| `--crcl`                | Use this flag to use the simplistic curriculum (see our paper for details) |