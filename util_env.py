import os
import yaml
import numpy as np
import rospy
import rospkg
import csv
import random
#import environments
from scipy.spatial.transform import Rotation

#from tf.transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import Point, Pose, PoseStamped
#from gazebo_msgs.msg import ModelState
#from gazebo_msgs.srv import GetLinkState
#from gazebo_msgs.srv import SetModelState
#from gazebo_msgs.srv import DeleteModel
#from gazebo_msgs.srv import SpawnModel

#from simulation.srv import VacuumGripperControl


def get_target_object(object_type='free_shapes'):
    # get list of target object
    # targetobj_fname = os.path.dirname('/home/adarshsehgal/workspace/ga-drl-aubo/newher/object_information.yaml')
    with open(('object_information.yaml'), "r") as stream:
        out = yaml.load(stream)
        filtered_object = list(filter(lambda x: x["type"] == object_type, out['items']))
        return filtered_object


def get_state(observation):
    """
    convert observation list into a numpy array
    """
    x = np.asarray(observation)
    return x

'''
def turn_on_gripper():
    """
    turn on the Gripper by calling the service
    """
    try:
        turn_on_gripper_service = rospy.ServiceProxy('/pickbot/gripper/control', VacuumGripperControl)
        enable = True
        turn_on_gripper_service(enable)
    except rospy.ServiceException as e:
        rospy.loginfo("Turn on Gripper service call failed:  {0}".format(e))


def turn_off_gripper():
    """
    turn off the Gripper by calling the service
    """
    try:
        turn_off_gripper_service = rospy.ServiceProxy('/pickbot/gripper/control', VacuumGripperControl)
        enable = False
        turn_off_gripper_service(enable)
    except rospy.ServiceException as e:
        rospy.loginfo("Turn off Gripper service call failed:  {0}".format(e))

'''
'''
def spawn_object(object_name, model_position, model_sdf=None):
    """
    spawn object using gazebo service
    :param object_name: name of the object
    :param model_position: position of the spawned object
    :param model_sdf: description of the object in sdf format
    :return: -
    """
    if model_sdf is None:  # take sdf file from default folder
        # get model from sdf file
        rospack = rospkg.RosPack()
        sdf_fname = rospack.get_path('simulation') + "/meshes/environments/sdf/" + object_name + ".sdf"
        with open(sdf_fname, "r") as f:
            model_sdf = f.read()

    try:
        spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        spawn_model(object_name, model_sdf, "/", model_position, "world")
        print("SPAWN %s finished" % object_name)
    except rospy.ServiceException as e:
        rospy.loginfo("Spawn Model service call failed:  {0}".format(e))


def spawn_urdf_object(object_name, model_position, model_urdf=None):
    """
    spawn object using gazebo service
    :param object_name: name of the object
    :param model_position: position of the spawned object
    :param model_sdf: description of the object in sdf format
    :return: -
    """
    if model_urdf is None:  # take sdf file from default folder
        # get model from sdf file
        rospack = rospkg.RosPack()
        urdf_fname = rospack.get_path('simulation') + "/urdf/" + object_name + ".urdf"
        with open(urdf_fname, "r") as f:
            model_urdf = f.read()

    try:
        spawn_model = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
        spawn_model(object_name, model_urdf, "/", model_position, "world")
        print("SPAWN %s finished" % object_name)
    except rospy.ServiceException as e:
        rospy.loginfo("Spawn Model service call failed:  {0}".format(e))


def delete_object(object_name):
    """
    delete object using gazebo service
    :param object_name: the name of the object
    :return: -
    """
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model(object_name)
    except rospy.ServiceException as e:
        rospy.loginfo("Delete Model service call failed:  {0}".format(e))
'''
'''
def change_object_position(object_name, model_position):
    """
    change object postion using gazebo service
    :param object_name: name of the object
    :param model_position: destination
    :return: -
    """
    try:
        change_position = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        box = ModelState()
        box.model_name = object_name
        box.pose.position.x = model_position.position.x
        box.pose.position.y = model_position.position.y
        box.pose.position.z = model_position.position.z
        box.pose.orientation.x = model_position.orientation[1]
        box.pose.orientation.y = model_position.orientation[2]
        box.pose.orientation.z = model_position.orientation[3]
        box.pose.orientation.w = model_position.orientation[0]
        change_position(box)
    except rospy.ServiceException as e:
        rospy.loginfo("Set Model State service call failed:  {0}".format(e))
'''
def get_target_position():
    '''
    try:
        #model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        linkNameTarget = "target"
        ReferenceFrame = "ground_plane"
        #object_resp_coordinates = model_coordinates(linkNameTarget, ReferenceFrame)
        #Object = np.array((object_resp_coordinates.link_state.pose.position.x, object_resp_coordinates.link_state.pose.position.y, object_resp_coordinates.link_state.pose.position.z))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get link state")
    '''

    Object = np.array((-0.3, 0.5, 0.2))
    return Object

def current_pos_moveit_callback(self, msg):
        self.curr_pos_moveit = msg

'''  
def get_distance_gripper_to_object(height=None):
    
    Get the Position of the endeffektor and the object via rosservice /gazebo/get_link_state
    Calculate distance between them
    In this case
    Object:     unite_box_0 link
    Gripper:    vacuum_gripper_link ground_plane
   
    
    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        linkNameTarget = "target"
        ReferenceFrame = "ground_plane"
        object_resp_coordinates = model_coordinates(linkNameTarget, ReferenceFrame)
        Object = np.array((object_resp_coordinates.link_state.pose.position.x, 
                            object_resp_coordinates.link_state.pose.position.y,
                            object_resp_coordinates.link_state.pose.position.z if height is None else object_resp_coordinates.link_state.pose.position.z + height/2))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get link state")

    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        LinkName = "vacuum_gripper_link"
        ReferenceFrame = "ground_plane"
        resp_coordinates_gripper = model_coordinates(LinkName, ReferenceFrame)
        Gripper = np.array((resp_coordinates_gripper.link_state.pose.position.x,
                            resp_coordinates_gripper.link_state.pose.position.y,
                            resp_coordinates_gripper.link_state.pose.position.z))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get Gripper position")
    distance = np.linalg.norm(Object - Gripper)
    
    Object = np.array((object_resp_coordinates.link_state.pose.position.x, 
                            object_resp_coordinates.link_state.pose.position.y,
                            object_resp_coordinates.link_state.pose.position.z if height is None else object_resp_coordinates.link_state.pose.position.z + height/2))

    return distance, Object
'''

def get_random_door_handle_pos():
    positions = [[-0.26, 0.848, 1.1, 0, 0, 1.57],
                 [-0.195, 0.848, 1.1, 0, 0, 1.57],
                 [-0.13, 0.848, 1.1, 0, 0, 1.57],
                 [-0.065, 0.848, 1.1, 0, 0, 1.57],
                 [0, 0.848, 1.1, 0, 0, 1.57],
                 [0.065, 0.848, 1.1, 0, 0, 1.57],
                 [0.13, 0.848, 1.1, 0, 0, 1.57],
                 [0.196, 0.848, 1.1, 0, 0, 1.57],
                 [-0.224, 0.995, 1.1, 0, 0, -1.57],
                 [-0.159, 0.995, 1.1, 0, 0, -1.57],
                 [-0.094, 0.995, 1.1, 0, 0, -1.57],
                 [-0.029, 0.995, 1.1, 0, 0, -1.57],
                 [0.036, 0.995, 1.1, 0, 0, -1.57],
                 [0.101, 0.995, 1.1, 0, 0, -1.57],
                 [0.166, 0.995, 1.1, 0, 0, -1.57],
                 [0.231, 0.995, 1.1, 0, 0, -1.57]]

    rand = random.choice(positions)
    rot = Rotation.from_euler('xyz', [rand[3], rand[4], rand[5]], degrees=True)
    random_door_pos = Pose(position=Point(x=rand[0], y=rand[1], z=rand[2]),
                           orientation=rot.as_quat())
    return random_door_pos


def append_to_csv(csv_filename, anarray):
    outfile = open(csv_filename, 'a')
    writer = csv.writer(outfile)
    writer.writerow(anarray)
    outfile.close()


def load_samples_from_prev_task(filename):  # currently for door handle only
    output = np.genfromtxt(filename, delimiter=',')
    action = output[:, 1:7]
    pos = output[:, 9:12]
    pos_orient = np.empty((0, 6))

    for i in range(len(pos)):
        _pos = pos[i]
        if pos[i, 1] > 0.99:
            _ori = [0, 0, -1.57]
        else:
            _ori = [0, 0, 1.57]
        _pos_ori = np.append(_pos, _ori, axis=0)
        pos_orient = np.append(pos_orient, [_pos_ori], axis=0)

    samples = np.append(action, pos_orient, axis=1)
    return samples

if __name__ == '__main__':
    print(get_target_object())
