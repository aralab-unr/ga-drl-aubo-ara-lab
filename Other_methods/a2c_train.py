import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
keys = ['observation', 'desired_goal']
#env = make_vec_env("FetchPush-v1", n_envs=4)
env = gym.make('FetchReach-v1', reward_type='dense')
env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation( env, keys))

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()