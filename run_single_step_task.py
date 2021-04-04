import os
import plot
import argparse
import pybullet_multigoal_gym as pmg
from drl_implementation import GoalConditionedDDPG
from DDPGHER_Params import params
parser = argparse.ArgumentParser()

parser.add_argument('--task', dest='task', type=str,
                    help='Name of the task to run', default='reach', choices=['reach', 'push', 'pick_and_place', 'slide'])
parser.add_argument('--num-runs', dest='num_runs', type=int,
                    help='Number of task runs, each of which runs with a unique random seed', default=4)
parser.add_argument('--render', dest='render',
                    help='Whether to render the task, default: False', default=False, action='store_true')
parser.add_argument('--joint-ctrl', dest='joint_ctrl',
                    help='Whether to use joint control mode, default: False', default=False, action='store_true')
parser.add_argument('--hindsight', dest='hindsight',
                    help='Whether to use hindsight experience replay, default: False', default=False, action='store_true')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    # random seeds
    seeds = [11 * (i+1) for i in range(args['num_runs'])]
    # hindsight experience replay
    params['hindsight'] = args['hindsight']
    # directory for storing data
    path = os.path.dirname(os.path.realpath(__file__))
    directory_name = args['task']
    if args['joint_ctrl']:
        directory_name += '_joint_ctrl'
    if args['hindsight']:
        directory_name += '_her'
    path = os.path.join(path, directory_name)

    seed_returns = []
    seed_success_rates = []
    for seed in seeds:
        # make env instance
        env = pmg.make_env(task=args['task'],
                           gripper='parallel_jaw',
                           joint_control=args['joint_ctrl'],
                           render=args['render'],
                           binary_reward=True,
                           max_episode_steps=50)

        seed_path = path + '/seed'+str(seed)

        agent = GoalConditionedDDPG(algo_params=params, env=env, path=seed_path, seed=seed)
        agent.run(test=False)
        seed_returns.append(agent.statistic_dict['epoch_test_return'])
        seed_success_rates.append(agent.statistic_dict['epoch_test_success_rate'])
        del env, agent

    return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                                   file_name=os.path.join(path, 'return_statistic.json'))
    plot.smoothed_plot_mean_deviation(path + '/returns', return_statistic, x_label='Epoch', y_label='Average returns')

    success_rate_statistic = plot.get_mean_and_deviation(seed_success_rates, save_data=True,
                                                         file_name=os.path.join(path, 'success_rate_statistic.json'))
    plot.smoothed_plot_mean_deviation(path + '/success_rates', success_rate_statistic,
                                      x_label='Epoch', y_label='Success rates')
