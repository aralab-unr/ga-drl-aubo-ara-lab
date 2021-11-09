#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelConfiguration
from gazebo_msgs.srv import SetModelConfigurationRequest

import math
import random
import numpy as np


class JointArrayPub(object):
    def __init__(self):
        self.joint_pub = rospy.Publisher('/pickbot/target_joint_positions', JointState, queue_size=10)
        self.relative_joint_pub = rospy.Publisher('/pickbot/relative_joint_positions', JointState, queue_size=10)
        self.init_pos = [1.5, -1.2, 1.4]

    def set_init_pose(self):
        """
        Sets joints to initial position
        :return:
        """
        # self.check_publishers_connection()
        self.pub_joints_to_moveit(self.init_pos)
        # self.reset_joints =  rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(100)  # 10hz
        while (self.joint_pub.get_num_connections() == 0):
            rospy.logdebug("No susbribers to _joint1_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("joint_pub Publisher Connected")

    def pub_joints_to_moveit(self, joints_array):
        self.check_publishers_connection()
        new_array = np.append(joints_array, [0.0, 0.0, 0.0])

        jointState = JointState()
        jointState.header = Header()
        jointState.header.stamp = rospy.Time.now()
        jointState.name = ['shoulder_joint', 'foreArm_joint', 'upperArm_joint',
                           'wrist1_joint', 'wrist2_joint', 'wrist3_joint']
        jointState.position = new_array
        jointState.velocity = []
        jointState.effort = []
        self.joint_pub.publish(jointState)
        # print("I've published: {}".format(jointState.position))

    def pub_relative_joints_to_moveit(self, joints_array):
        self.check_publishers_connection()
        new_array = np.append(joints_array, [0.0, 0.0])
        jointState = JointState()
        jointState.header = Header()
        jointState.header.stamp = rospy.Time.now()
        jointState.name = ['shoulder_joint', 'foreArm_joint', 'upperArm_joint',
                           'wrist1_joint', 'wrist2_joint', 'wrist3_joint']
        jointState.position = new_array
        jointState.velocity = []
        jointState.effort = []
        self.relative_joint_pub.publish(jointState)

    def set_joints(self, array=[1.5, -1.2, 1.4, 0.0]):
        # reset_req = SetModelConfigurationRequest()
        # reset_req.model_name = 'pickbot'
        # reset_req.urdf_param_name = 'robot_description'
        # reset_req.joint_names =[ 'elbow_joint', 'shoulder_lift_joint','shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        # reset_req.joint_positions = array
        # res = self.reset_joints(reset_req)
        self.pub_joints_to_moveit(array)


if __name__ == "__main__":
    rospy.init_node('joint_array_publisher_node', anonymous=True)
    joint_publisher = JointArrayPub()