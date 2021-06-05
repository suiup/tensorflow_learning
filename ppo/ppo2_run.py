
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import numpy as np

# multiprocess environment
env = make_vec_env('CartPole-v1', n_envs=2)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
obs = env.reset()
index = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    index += 1
    if(all(dones)):
        print("index: ", index)
        print(f"state : {obs}, reward : {rewards}")
        break
