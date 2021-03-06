"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import os
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

SavedState = namedtuple("SavedState", "state_dict timestep stats")
# STATISTICS_FILE_PATH = 'statistics.pkl'
print('******* Running on {} *******'.format('CUDA' if USE_CUDA else 'CPU'))


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    feature_tested,
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    ):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete
    # added bool_flag for double save statistic
    bool_flag = False
    STATS_FILE_NAME = 'statistics ' + feature_tested + '.pkl'

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            with torch.no_grad():
                # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
                return model(Variable(obs)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function, i.e. build the model.
    ######

    # YOUR CODE HERE

    if USE_CUDA:
        Q = q_func(num_actions=num_actions).cuda()
        Q_target = q_func(num_actions=num_actions).cuda()
    else:
        Q = q_func(num_actions=num_actions)
        Q_target = q_func(num_actions=num_actions)

    Q_target.load_state_dict(Q.state_dict())

    # Check & load pretrained model
    if os.path.isfile('Q_params'+feature_tested+'.pkl'):
        print('Load Q parameters ...')
        Q.load_state_dict(torch.load('Q_params'+feature_tested+'.pkl'))

    if os.path.isfile('target_Q_params'+feature_tested+'.pkl'):
        print('Load target Q parameters ...')
        Q_target.load_state_dict(torch.load('target_Q_params'+feature_tested+'.pkl'))

    ######


    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    Statistic = {
        "starting_Q_values": [],
        "mean_episode_rewards": [],
        "best_mean_episode_rewards": []
    }

    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')

    # load prev Stats
    start = 0
    if os.path.isfile(STATS_FILE_NAME):
        with open(STATS_FILE_NAME, 'rb') as f:
            Statistic = pickle.load(f)
            mean_episode_reward = Statistic["mean_episode_rewards"][-1]
            best_mean_episode_reward = Statistic["best_mean_episode_rewards"][-1]
            start = len(Statistic["mean_episode_rewards"])
            print('Load %s ...' % STATS_FILE_NAME)
    done = False

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 50000 #10000

    for t in count(start):
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            return Statistic
            break
        # couldnt handle stopping_criterion, this works:
        if t > 4500000:
            return Statistic
            break
        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)
        #####

        # YOUR CODE HERE
        idx = replay_buffer.store_frame(last_obs)
        encoded_obs = replay_buffer.encode_recent_observation()
        action = select_epilson_greedy_action(Q, encoded_obs, t)
        # if started a new game - store Q-func
        ######
        # done not initialized since last time

        if t > learning_starts and done:
            # a very expensive statistic - so don't log frequently
            with torch.no_grad():
                obs = torch.from_numpy(encoded_obs).type(dtype).unsqueeze(0) / 255.0
                item = torch.max(Q(Variable(obs))).item()
            Statistic["starting_Q_values"].append(item)

        ######
        # this steps the environment forward one step
        last_obs, reward, done, info = env.step(action)
        replay_buffer.store_effect(idx, action, reward, done)

        if done:
            last_obs = env.reset()

        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # Note: Move the variables to the GPU if avialable
            # 3.b: fill in your own code to compute the Bellman error. This requires
            # evaluating the current and next Q-values and constructing the corresponding error.
            # Note: don't forget to clip the error between [-1,1], multiply is by -1 (since pytorch minimizes) and
            #       maskout post terminal status Q-values (see ReplayBuffer code).
            # 3.c: train the model. To do this, use the bellman error you calculated perviously.
            # Pytorch will differentiate this error for you, to backward the error use the following API:
            #       current.backward(d_error.data.unsqueeze(1))
            # Where "current" is the variable holding current Q Values and d_error is the clipped bellman error.
            # Your code should produce one scalar-valued tensor.
            # Note: don't forget to call optimizer.zero_grad() before the backward call and
            #       optimizer.step() after the backward call.
            # 3.d: periodically update the target network by loading the current Q network weights into the
            #      target_Q network. see state_dict() and load_state_dict() methods.
            #      you should update every target_update_freq steps, and you may find the
            #      variable num_param_updates useful for this (it was initialized to 0)
            #####

            # YOUR CODE HERE
            # 3.a sample a batch of transitions
            sample = replay_buffer.sample(batch_size)
            obs_batch, action_batch, reward_batch, next_obs_batch, done_mask = sample

            # move variables to GPU if available

            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype)) / 255.0
            action_batch = Variable(torch.from_numpy(action_batch).type(dtype).long().view(-1, 1))
            reward_batch = Variable(torch.from_numpy(reward_batch).type(dtype))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype)) / 255.0
            done_mask = Variable(torch.from_numpy(done_mask).type(dtype))

            # 3.b compute the Bellman error
            # evaluating the current and next Q-values
            state_action_values = Q(obs_batch).gather(1, action_batch)
            next_state_values = Q_target(next_obs_batch).detach()

            # maskout post terminal status Q-values
            masked_next_state_values = next_state_values.max(1)[0] * (1 - done_mask)
            # constructing the corresponding error
            expected_state_action_values = (masked_next_state_values * gamma) + reward_batch
            bellman_error = expected_state_action_values.unsqueeze(1) - state_action_values

            # clip the error between [-1,1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            optimizer.zero_grad()
            # multiply by -1 (since pytorch minimizes)
            state_action_values.backward(-clipped_bellman_error)

            # 3.c: train the model
            optimizer.step()

            # 3.d periodically update the target network
            num_param_updates += 1
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            #####

        ### 4. Log progress and keep track of statistics
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()


            # Save the trained model
            torch.save(Q.state_dict(), 'Q_params'+feature_tested+'.pkl')
            torch.save(Q_target.state_dict(), 'target_Q_params'+feature_tested+'.pkl')
            # Dump statistics to pickle
            #double save
            if bool_flag:
                bool_flag = False
                with open(STATS_FILE_NAME, 'wb') as f:
                    pickle.dump(Statistic, f)
                    print("Saved to %s" % STATS_FILE_NAME)
            else:
                bool_flag = True
                with open('copy_'+STATS_FILE_NAME, 'wb') as f:
                    pickle.dump(Statistic, f)
                    print("Saved to %s" % 'copy_'+STATS_FILE_NAME)


            plt.clf()
            plt.xlabel('Num of Games')
            plt.ylabel('Q-values on starting state')
            plt.plot(range(len(Statistic["starting_Q_values"])), Statistic["starting_Q_values"], label='Q-values')
            plt.legend()
            plt.title(feature_tested)
            plt.savefig('Q-value-Performance'+feature_tested+'.png')

            plt.clf()
            plt.xlabel('Timesteps')
            plt.ylabel('Mean Reward (past 100 episodes)')
            num_items = len(Statistic["mean_episode_rewards"])
            plt.plot(range(num_items), Statistic["mean_episode_rewards"], label='mean reward')
            plt.plot(range(num_items), Statistic["best_mean_episode_rewards"], label='best mean rewards')
            plt.legend()
            plt.title(feature_tested)
            plt.savefig('DeepQ-Performance'+feature_tested+'.png')
            # plt.show()

