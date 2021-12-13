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
    id='AuboReach-v5',
    entry_point='aubo_reach8_env:PickbotEnv',
    max_episode_steps=200,  # 100
)


# DEFINE ENVIRONMENT CLASS
class PickbotEnv(gym.GoalEnv):

    def __init__(self, joint_increment=None, sim_time_factor=0.005, random_object=False, random_position=False,
                 use_object_type=False, populate_object=False, env_object_type='free_shapes'):

        self.added_reward = 0
        self.seed()
        self.count = 0
        self.rewardThreshold = 0.80
        self.new_action = [0., 0., 0., 0.]
        # self.init_pos = np.array([.3,0.7,-0.2,1.3])
        self.init_pos = np.array([-1.3, 0.4, 1.2, -1.0])
        self.action_shape = 4
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype="float32")

        # self.goal = np.array([-0.503, 0.605, -1.676])
        # self.goal = np.array([-0.503, 0.605, -1.676, 1.391])
        self.goal = np.array([0.612, 1.3566, -1.234, 0.4995])
        # self.goal = np.array([0.5,1.2,1.4,-1.5])

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

    def random_init_joints(self):
        for i in range(len(self.init_pos)):
            self.init_pos[i] = np.random.default_rng().uniform(low=-1.7, high=1.7)

    def random_goal(self):
        goal_joints = np.zeros(self.action_shape)

        for i in range(len(goal_joints)):
            goal_joints[i] = np.random.default_rng().uniform(low=-1.7, high=1.7)

        return goal_joints

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        return -d
        # return -(d > 0.1).astype(np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        row_list = [self.counter, self.added_reward]
        with open('rewards.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(row_list)
            self.counter = self.counter + 1

        self.added_reward = 0
        # self.init_pos = [.3,0.7,-0.2,1.3]
        self.random_init_joints()
        self.goal = self.random_goal()
        # print('random init: ', self.init_pos)
        # print('random goal: ', self.goal)
        # self.new_action = [0,0,0]
        observation = self._get_obs()
        # self.init_pos = np.array([.3,0.7,-0.2,1.3])
        return observation

    def step(self, action):

        # print('=======================next_action=====================')
        # print(action)

        self.new_action = action.copy()
        # print(action)
        # print(self.new_action)
        assert self.new_action[0] == action[0]

        obs = self._get_obs()
        # assert self.new_action[0] == obs["achieved_goal"][0]
        # print(obs["achieved_goal"])
        # print('=============================================')

        done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        # print(reward)
        self.added_reward += reward
        # print(obs["achieved_goal"])
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
        wrist_joint = self.new_action[3]
        # four = self.new_action[3]

        curr_joint = np.array(
            [shoulder_joint_state, foreArm_joint_state, upperArm_joint_state, wrist_joint])
        object = self.goal
        spot_counter = 0
        curr_joint_increment = curr_joint * 0.05
        for i in curr_joint:
            self.init_pos[spot_counter] += curr_joint_increment[spot_counter]
            spot_counter += 1

        # print(self.init_pos)
        a_goal = self.init_pos.copy()
        # print(achieved_goal)
        achieved_goal = np.asarray(a_goal)
        # print(achieved_goal)
        rel_pos = achieved_goal - object
        relative_pose = rel_pos
        obs = np.concatenate([achieved_goal, relative_pose])
        # print(achieved_goal)
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

        return (d < 0.1).astype(np.float32)
        # return d