import gym
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
import random

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule, ConstantSchedule


BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

"""
# deepmind's hyperparameters:
    exploration_schedule = LinearSchedule(1000000, 0.1)
    RMSProp algorithm with minibatches of size 32.
"""


def my_load(path):
    with open(path, 'rb') as f:
        saved_state = pickle.load(f)
    return saved_state
    # load previous data


"""
    if os.path.isfile(save_path):
        saved_state = my_load(save_path + feature_tested)
        Q.load_state_dict(saved_state.state_dict)
        Q_target.load_state_dict(saved_state.state_dict)
        start = saved_state.timestep
        Statistic = saved_state.stats
        mean_episode_reward = Statistic["mean_episode_rewards"][-1][1]
        best_mean_episode_reward = Statistic["best_mean_episode_rewards"][-1][1]
"""


def main(env, num_timesteps):
    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = random.randint(0,100)  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    # empty dict to hold all results
    Stats = {}

    new_lr = 0.001
    new_gamma = 0.999
    exploration_sches = [LinearSchedule(1000000, 0.1), ConstantSchedule(0.05),
                         ConstantSchedule(0.15), LinearSchedule(500000, 0.05)]

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=new_lr, alpha=ALPHA, eps=EPS),
    )

    env = get_env(task, seed)
    Stats["lr=0.001, gamma=0.999"] = dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=new_gamma,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        feature_tested="lr=0.001, gamma=0.999"
    )

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    env = get_env(task, seed)
    Stats["Default"] = dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        feature_tested=""
    )

    plt.clf()
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward (past 100 episodes)')
    num_items = len(Stats["lr=0.001, gamma=0.999"]["mean_episode_rewards"])
    plt.plot(range(num_items), Stats["lr=0.001, gamma=0.999"]["mean_episode_rewards"], label="lr=0.001, gamma=0.999")
    num_items = len(Stats["Default"]["mean_episode_rewards"])
    plt.plot(range(num_items), Stats["Default"]["mean_episode_rewards"], label="Default")
    plt.legend()
    plt.title("Performance")
    plt.savefig('Final-Performance.png')


if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0  # datetime.now()  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env, task.max_timesteps)
