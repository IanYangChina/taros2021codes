import os
import plot
import argparse
import pybullet_multigoal_gym as pmg
from drl_implementation import GoalConditionedDDPG
from DDPGHER_Params import params
parser = argparse.ArgumentParser()

parser.add_argument('--task', dest='task', type=str,
                    help='Name of the task to run', default='block_rearrange', choices=['block_rearrange', 'block_stack', 'chest_pick_and_place', 'chest_push'])
parser.add_argument('--num-runs', dest='num_runs', type=int,
                    help='Number of task runs, each of which runs with a unique random seed', default=4)
parser.add_argument('--num-blocks', dest='num_blocks', type=int,
                    help='Number of blocks', default=2, choices=[1, 2, 3, 4, 5])
parser.add_argument('--render', dest='render',
                    help='Whether to render the task, default: False', default=False, action='store_true')
parser.add_argument('--crcl', dest='crcl',
                    help='Whether to use the simplistic curriculum (see our paper for details), default: False', default=False, action='store_true')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    # random seeds
    seeds = [11 * (i+1) for i in range(args['num_runs'])]
    # adjust training epochs and maximum episode timesteps
    if args['task'] in ['block_rearrange', 'block_stack']:
        assert args['num_blocks'] > 1, 'the number of blocks for block rearrange or stack tasks should be greater than 1'
        params['training_epochs'] = (args['num_blocks'] - 1) * 50 + 1
        max_episode_steps = 50 + (args['num_blocks'] - 1) * 25
    else:
        params['training_epochs'] = 50 + (args['num_blocks'] - 1) * 50 + 1
        max_episode_steps = 50 + (args['num_blocks']) * 25
    # update the absolute clip value
    params['clip_value'] = max_episode_steps
    # calculate the total training episodes for curriculum generation
    num_total_episodes = params['training_epochs']*params['training_cycles']*params['training_episodes']
    # directory for storing data
    path = os.path.dirname(os.path.realpath(__file__))
    directory_name = args['task'] + '_' + str(args['num_blocks'])
    params['curriculum'] = args['crcl']
    if args['crcl']:
        directory_name += '_crcl'
    path = os.path.join(path, directory_name)

    seed_returns = []
    seed_success_rates = []
    for seed in seeds:
        # make env instance
        env = pmg.make_env(task=args['task'],
                           gripper='parallel_jaw',
                           num_block=args['num_blocks'],
                           render=args['render'],
                           binary_reward=True,
                           image_observation=False,
                           use_curriculum=args['crcl'],
                           num_goals_to_generate=num_total_episodes,
                           max_episode_steps=max_episode_steps)

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
