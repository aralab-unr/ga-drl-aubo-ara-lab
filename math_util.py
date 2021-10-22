import math
import numpy as np


#def compute_reward(observation, done_reward, invalid_contact):
def compute_reward(observation, done_reward):
    """
    Calculates the reward in each Step
    Reward for:
    Distance:       Reward for Distance to the Object
    Contact:        Reward for Contact with one contact sensor and invalid_contact must be false. As soon as both
                    contact sensors have contact and there is no invalid contact the goal is considered to be reached
                    and the episode is over. Reward is then set in is_done
    Calculates the Reward for the Terminal State
    Done Reward:    Reward when episode is Done. Negative Reward for Crashing and going into set Joint Limits.
                    High positive reward for having contact with both contact sensors and not having an invalid collision
    """

    ####################################################################
    #                        Plan0: init                               #
    ####################################################################

    # reward_contact = 0
    #
    # # Reward for Distance to encourage approaching the target object
    # distance = observation[0]
    # # reward_distance = 1 - math.pow(distance / max_distance, 0.4)
    # relative_distance = observation[-1] - distance
    # reward_distance = relative_distance * 20 if relative_distance < 0 else relative_distance * 10
    #
    # # Reward for Contact
    # contact_1 = observation[7]
    # contact_2 = observation[8]
    #
    # if contact_1 == 0 and contact_2 == 0:
    #     reward_contact = 0
    # elif contact_1 != 0 and contact_2 == 0 and not invalid_contact or contact_1 == 0 and contact_2 != 0 and \
    #         not invalid_contact:
    #     reward_contact = 2000
    #     reward_distance = 0
    #
    # total_reward = reward_distance + reward_contact + done_reward
    #
    # print("reward_distance: {}".format(reward_distance))
    #
    # return total_reward

    ####################################################################################
    # Plan1: Reach a point in 3D space (usually right above the target object)         #
    # Reward only dependent on distance. Nu punishment for crashing or joint_limits    #
    ####################################################################################
    new_obs = observation['observation']
    distance = new_obs[0]
    x = np.sqrt((1/3)) * distance
    alpha = 5
    done = 0.02
    a = np.exp(-alpha*x) - np.exp(-alpha) + 10 * (np.exp(-alpha*x / done) - np.exp(-alpha))
    b = 1 - np.exp(-alpha)
    reward_distance = a/b - 1
    print("reward_distance: {}".format(reward_distance))

    total_reward = reward_distance + done_reward

    return total_reward


def compute_reward_orient(observation, done_reward, invalid_contact):
    """
    Calculates the reward in each Step
    Reward for:
    Distance:       Reward for Distance to the Object
    Contact:        Reward for Contact with one contact sensor and invalid_contact must be false. As soon as both
                    contact sensors have contact and there is no invalid contact the goal is considered to be reached
                    and the episode is over. Reward is then set in is_done
    Calculates the Reward for the Terminal State
    Done Reward:    Reward when episode is Done. Negative Reward for Crashing and going into set Joint Limits.
                    High positive reward for having contact with both contact sensors and not having an invalid collision
    """

    # Reward for Distance to encourage approaching the box
    distance = observation[0]
    # reward_distance = 1 - math.pow(distance / max_distance, 0.4)
    relative_distance = observation[-2] - distance
    reward_distance = relative_distance * 20 if relative_distance < 0 else relative_distance * 10

    # Reward for orientation
    orient_differences = observation[-1]

    reward_orient = 0
    if not invalid_contact:
        reward_orient = (1 - orient_differences/math.pi) * 10

    total_reward = reward_distance * reward_orient + done_reward

    print("distance: {} orient:{} total:{}".format(reward_distance, reward_orient, total_reward))

    return total_reward


def rmseFunc(eePoints):
    """
    Computes the Residual Mean Square Error of the difference between current and desired
     end-effector position
    """
    rmse = np.sqrt(np.mean(np.square(eePoints), dtype=np.float32))
    return rmse


#def computeReward(rewardDist, rewardOrientation=0, collision=False):
def computeReward(rewardDist, rewardOrientation=0):
    alpha = 5
    beta = 1.5
    gamma = 1
    delta = 3
    eta = 0.03
    done = 0.02

    distanceReward = (math.exp(-alpha * rewardDist) - math.exp(-alpha)) \
     / (1 - math.exp(-alpha)) + 10 * (math.exp(-alpha/done * rewardDist) - math.exp(-alpha/done)) \
     / (1 - math.exp(-alpha/done))
    orientationReward = (1 - (rewardOrientation / math.pi)**beta + gamma) / (1 + gamma)
    '''
    if collision:
        rewardDist = min(rewardDist, 0.5)
        collisionReward = delta * (2 * rewardDist)**eta
    else:
        collisionReward = 0
    '''
    
    print("Reward distance {} orientation {}".format(distanceReward, orientationReward))

    #return distanceReward * orientationReward - 1 - collisionReward
    return distanceReward * orientationReward - 1