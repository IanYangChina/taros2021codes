import time
import gym
import plot
import numpy as np
import pybullet_multigoal_gym as pmg

seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 100]

costs_mujoco = []
costs_pmg = []

num_episodes = 100

for seed in seeds:

    # mujoco loop
    start_mujoco = time.time()

    env_mujoco = gym.make("FetchReach-v1")
    env_mujoco.seed(seed)

    for i in range(num_episodes):
        env_mujoco.reset()
        done_mujoco = False
        while not done_mujoco:
            action = env_mujoco.action_space.sample()
            _, _, done_mujoco, _ = env_mujoco.step(action)

    cost = time.time() - start_mujoco
    costs_mujoco.append(cost/num_episodes)

    # pmg loop
    start_pmg = time.time()

    env_pmg = pmg.make_env(task='reach',
                           gripper='parallel_jaw',
                           joint_control=True,
                           render=True,
                           binary_reward=True,
                           max_episode_steps=50,
                           image_observation=False,
                           depth_image=False,
                           goal_image=False)
    env_pmg.seed(seed)

    for i in range(num_episodes):
        env_pmg.reset()
        done_pmg = False
        while not done_pmg:
            action = env_pmg.action_space.sample()
            _, _, done_pmg, _ = env_pmg.step(action)

    cost = time.time() - start_pmg
    costs_pmg.append(cost/num_episodes)

mean_costs_mujoco = np.mean(costs_mujoco)
std_costs_mujoco = np.std(costs_mujoco)
mean_costs_pmg = np.mean(costs_pmg)
std_costs_pmg = np.std(costs_pmg)

print("Computation costs (in seconds) of running the Reach task for 1 episode, "
      "averaged over 100 episodes for 10 random seeds:"
      "\\ \\ Mujoco: mean {} std {}\n"
      "\\ \\ PMG: mean {} std {}".format(mean_costs_mujoco, std_costs_mujoco, mean_costs_pmg, std_costs_pmg))