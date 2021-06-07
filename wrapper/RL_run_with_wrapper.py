import gym
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack

from normalize_actions_wrapper import NormalizeActionWrapper

normalized_env = Monitor(gym.make('Pendulum-v0'), filename=None, allow_early_resets=True)
# Note that we can use multiple wrappers
normalized_env = NormalizeActionWrapper(normalized_env)
normalized_env = DummyVecEnv([lambda: normalized_env])

model_2 = A2C("MlpPolicy", normalized_env, verbose=1).learn(int(1000))

print("----------------------------------------------------------")

env = DummyVecEnv([lambda: gym.make("Pendulum-v0")])
normalized_vec_env = VecNormalize(env)

obs = normalized_vec_env.reset()
for _ in range(10):
  action = [normalized_vec_env.action_space.sample()]
  obs, reward, _, _ = normalized_vec_env.step(action)
  print(obs, reward)