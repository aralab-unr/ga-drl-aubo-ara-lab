#!/usr/bin/env python

# IMPORT
import gym
import rospy
import numpy as np
import time
import random
import sys
import yaml
import math
import datetime
import csv
import rospkg
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register

# OTHER FILES
# import util_env as U
# import math_util as UMath
# from joint_array_publisher import JointArrayPub
import logger

# MESSAGES/SERVICES


register(
    id='AuboReach-v2',
    entry_point='aubo_reach5_env:PickbotEnv',
    max_episode_steps=50,  # 100
)


# DEFINE ENVIRONMENT CLASS
class PickbotEnv(gym.GoalEnv):

    def __init__(self, joint_increment=None, sim_time_factor=0.005, random_object=False, random_position=False,
                 use_object_type=False, populate_object=False, env_object_type='free_shapes'):
        self.added_reward = 0
        self.seed()

        self.rewardThreshold = 0.1
        self.new_action = [0, 0, 0, 0, 0, 0]

        self.action_space = spaces.Box(-1.7, 1.7, shape=(6,), dtype="float32")

        # self.goal = np.array([-0.503, 0.605, -1.676])
        self.goal = np.array([-0.503, 0.605, -1.676, 1.367, -1.527, -0.036])

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )
        # self.reward_range = (-np.inf, np.inf)

        self.counter = 0

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        return -(d > self.rewardThreshold).astype(np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        row_list = [self.counter, self.added_reward]
        # with open('rewards.csv', 'a', encoding='UTF8', newline='') as f:
        #     writer = csv.writer(f)
        #
        #     # write the header
        #     writer.writerow(row_list)
        #     self.counter = self.counter + 1

        self.added_reward = 0

        # self.new_action = [0,0,0]
        observation = self._get_obs()

        return observation

    def step(self, action):
        # print('=======================next_action=====================')
        # print(action)
        self.new_action = action.copy()
        # print(self.new_action)
        assert self.new_action[0] == action[0]
        obs = self._get_obs()
        assert self.new_action[0] == obs["achieved_goal"][0]
        # print(obs["achieved_goal"])
        # print('=============================================')

        done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        # print(reward)
        self.added_reward += reward

        # info = {}

        row_list = [reward, self.counter]
        # with open('rewards.csv', 'a', encoding='UTF8', newline='') as f:
        #   writer = csv.writer(f)

        # write the header
        #  writer.writerow(row_list)
        #  self.counter = self.counter + 1
        return obs, reward, done, info

    def _get_obs(self):
        shoulder_joint_state = self.new_action[0]
        foreArm_joint_state = self.new_action[1]
        upperArm_joint_state = self.new_action[2]
        four = self.new_action[3]
        wrist2_joint_state = self.new_action[4]
        wrist3_joint_state = self.new_action[5]

        curr_joint = np.array(
            [shoulder_joint_state, foreArm_joint_state, upperArm_joint_state, four, wrist2_joint_state, wrist3_joint_state])
        object = self.goal

        achieved_goal = curr_joint.copy()
        rel_pos = achieved_goal - object
        relative_pose = rel_pos
        obs = np.concatenate([achieved_goal, relative_pose])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        # calc_d = 1 - (0.12 + 0.88 * (d / 10))
        # calc_d = 1 - d
        # self.calculatedReward = calc_d

        return (d < self.rewardThreshold).astype(np.float32)
        # return d
