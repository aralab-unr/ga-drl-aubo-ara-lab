import os
import sys

import click
import numpy as np
import json
import sys
import csv
from mpi4py import MPI
import datetime
import logger
from misc_util import set_global_seeds
from mpi_moments import mpi_moments
import config as config
from rollout import RolloutWorker
from util import mpi_fork

total_success = 0
average_success = 0
epoch_to_save = 0
is_not_done = True
from subprocess import CalledProcessError


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    begin_time = datetime.datetime.now()


    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):            
            #logger.info(config.DEFAULT_PARAMS['_polyak'])
            #config.DEFAULT_PARAMS['_polyak'] = round(random.uniform(0, 1), 3)
            #logger.info('polyak is :')
            episode = rollout_worker.generate_rollouts()
            #print(episode['o'][:, 1:, :].shape)
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            row_list1 = [key, mpi_average(val)]
            with open('fetch_push_vals.csv', 'a', encoding='UTF8', newline= '') as f:
                writer = csv.writer(f)
                writer.writerow(row_list1)
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        

        if rank == 0:
            logger.dump_tabular()

        
        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        global total_success
        #average_success = 0
        #checking if success rate has reached close to maximum, if so, return number of epochs
        global average_success
        global epoch_to_save
        global is_not_done
        average_success += success_rate

        if success_rate >= 0.9:
            total_success += 1
        else:
            total_success = 0
        


        if total_success >= 3: 
            logger.info('Saving epochs to file...')
            with open('epochs.txt', 'w') as output:
                output.write(str(epoch+1))
            #Exit training if maximum success rate reached
            
            if is_not_done:

                row_list1 = [str(datetime.datetime.now() - begin_time)]
                with open('aubo_time.csv', 'a', encoding='UTF8', newline= '') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_list1)
                is_not_done = False
                
            epoch_to_save = epoch+1
            to_save = average_success / epoch_to_save
            row_list1 = [epoch_to_save, to_save]

            with open('ga_success.csv', 'a', encoding='UTF8', newline= '') as f:
                writer = csv.writer(f)
                writer.writerow(row_list1)
            
            sys.exit()
        if epoch==(n_epochs-1):
            logger.info('Maximum success rate not reached. Saving maximum epochs to file...')
            with open('epochs.txt', 'w') as output:
                output.write(str(n_epochs))
            row_list1 = [str(datetime.datetime.now() - begin_time)]
            if is_not_done:
                with open('aubo_time.csv', 'a', encoding='UTF8', newline= '') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_list1)
            epoch_to_save = n_epochs

            to_save = average_success / epoch_to_save
            row_list1 = [epoch_to_save, to_save]

            with open('ga_success.csv', 'a', encoding='UTF8', newline= '') as f:
                writer = csv.writer(f)
                writer.writerow(row_list1)

            sys.exit()   

        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, polyak_value, gamma_value, q_learning, pi_learning, random_epsilon, noise_epsilon,
    override_params={}, save_policies=True,
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    #print('DEFAULT_PARAMS ARE')
    #print(config.DEFAULT_PARAMS)
    params['env_name'] = env
    params['polyak'] = polyak_value
    params['gamma'] = gamma_value
    params['Q_lr'] = q_learning
    params['pi_lr'] = pi_learning
    params['random_eps'] = random_epsilon
    params['noise_eps'] = noise_epsilon
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)

    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    epochs = train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies)

    


    return epochs


@click.command()
@click.option('--env', type=str, default='FetchReacher-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default='/tmp/newlog', help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=40, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=4, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--polyak_value', type=float, default=0.95, help='polyak averaging coefficient - Tau')
@click.option('--gamma_value', type=float, default=0.98, help='gamma - discounting factor')
@click.option('--q_learning', type=float, default=0.001, help='critic learning rate')
@click.option('--pi_learning', type=float, default=0.001, help='actor learning rate')
@click.option('--random_epsilon', type=float, default=0.3, help='percentage of time a random action is taken')
@click.option('--noise_epsilon', type=float, default=0.2, help='std of gaussian noise added to not-completely-random actions as a percentage of max_u')
def main(**kwargs):
    launch(**kwargs)

if __name__ == '__main__':
    main()