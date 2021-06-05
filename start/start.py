from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

model = PPO2(MlpPolicy, 'CartPole-v1', verbose=1).learn(1000)
