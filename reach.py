import os
from gym import utils
import fetch
from gym.envs.registration import register

# Ensure we get the path separator correct on windows
#MODEL_XML_PATH = os.path.join("assets", "aubo_i5.xml")
MODEL_XML_PATH = 'aubo_i5.xml'
register(
            id="FetchReacher-v1",
            entry_point="reach:FetchReachEnv",
            max_episode_steps=50,
        )

class FetchReachEnv(fetch.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "shoulder_joint": 0.4,
            "upperArm_joint": 0.1,
            "foreArm_joint": 0.0,
        }
        fetch.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.1,
            distance_threshold=0.09,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)