import gym
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.remove('/usr/local/lib/python2.7/dist-packages')
import pybullet_envs

#from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
#from stable_baselines import PPO2
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

# Parallel environments
keys = ['observation', 'desired_goal']
#env = make_vec_env("FetchPush-v1", n_envs=4)
env = gym.make('FetchReach-v1', reward_type='dense')
env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation( env, keys))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    
    env.render()