import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import numpy as np
from stable_baselines.common.cmd_util import make_vec_env
import time
from env import taxi_env
from stable_baselines.common import set_global_seeds
import os
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

def evaluate_multi_processes(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    env = model.get_env()
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        actions, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(actions)
        # Stats
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]
            if dones[i]:
                episode_rewards[i].append(0.0)

        mean_rewards = [0.0 for _ in range(env.num_envs)]
        n_episodes = 0
        for i in range(env.num_envs):
            mean_rewards[i] = np.mean(episode_rewards[i])
            n_episodes += len(episode_rewards[i])

        # Compute mean reward
        mean_reward = round(np.mean(mean_rewards), 1)
        print("Mean reward:", mean_reward, "Num episodes:", n_episodes)
        return mean_reward

def make_env(env, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    n_steps = 4096
    batches = 256
    n_timesteps = 25000
    # multiprocess environment
    env_id = gym.Env
    # env_id = "Taxi-v3"
    # env_id = taxi_env.TaxiEnv
    # env_id = "CartPole-v1"
    num_cpu = 2  # Number of processes to use
    env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
    model = PPO2(MlpPolicy, env,n_steps=n_steps,nminibatches=batches, verbose=1)

    # Evaluate the un-trained, random agent
    # evaluate_multi_processes(model, num_steps=1000)

    start_time = time.time()
    model.learn(total_timesteps = n_timesteps)
    total_time_multi = time.time() - start_time
    print("Took {:.2f}s for multi-processed version - {:.2f} FPS".format(total_time_multi,
                                                                          n_timesteps/ total_time_multi))
    # model.save("ppo2_multi_env_cartpole")


    # Single Process RL Training
    # single_process_model = PPO2(MlpPolicy, DummyVecEnv([lambda: gym.make(env_id)]), verbose=0)
    single_process_model = PPO2(MlpPolicy, make_vec_env(env_id),n_steps=n_steps,nminibatches=batches, verbose=0)
    start_time = time.time()
    single_process_model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time_single,
                                                                        n_timesteps / total_time_single))

    print("Multi-processed training is {:.2f}x faster!".format(total_time_single / total_time_multi))
