# GA-DRL algorithm with robotic manipulator Aubo i5

# GA-DRL paper citation
```
@inproceedings{sehgal2019deep,
  title={Deep reinforcement learning using genetic algorithm for parameter optimization},
  author={Sehgal, Adarsh and La, Hung and Louis, Sushil and Nguyen, Hai},
  booktitle={2019 Third IEEE International Conference on Robotic Computing (IRC)},
  pages={596--601},
  year={2019},
  organization={IEEE}
}
```

## Prerequisite
- Must have compiled the aubo robot github repo under the kinetic branch,which can be found here:
  - It is safe to remove auto_controller folder if you get build error with this package
```
https://github.com/adarshsehgal/aubo_robot
```
- Ubuntu 16.04
- Ros Kinetic
- Python 2.7
- Python verison >= 3.5 (this code was run on python 3.5)
- Aubo gym environment uses python2.7 with moveit
- Genetic Algorithm ga.py uses python3.7
- pip install gym==0.15.6
- Install the packages needed to install gym
```
pip3 install scipy tqdm joblib cloudpickle click opencv-python
```
- pip install tensorflow==1.14.0
- openai_ros
  - IMPORTANT: run rosdep install openai_ros EACH time you run the code (for each terminal)
```
https://github.com/adarshsehgal/openai_ros
```
- update pip3 to 21.0 or latest (if no errors)
```
pip3 install --upgrade pip
```
- To avoid libmoveit_robot_trajectory.so error, follow below commands
  - replace the version number of 0.9.17 with what you have in below directory
  - don’t change 0.9.15 
```
cd /opt/ros/kinetic/lib 
sudo cp -r libmoveit_robot_trajectory.so.0.9.17 .. 
sudo cp -r libmoveit_robot_state.so.0.9.17 ..
cd .. 
sudo mv libmoveit_robot_state.so.0.9.17 libmoveit_robot_state.so.0.9.15 
sudo mv libmoveit_robot_model.so.0.9.17 libmoveit_robot_model.so.0.9.15
sudo mv libmoveit_robot_trajectory.so.0.9.17 libmoveit_robot_trajectory.so.0.9.15
sudo cp -r libmoveit_robot_state.so.0.9.15 lib/ 
sudo cp -r libmoveit_robot_model.so.0.9.15 lib/ 
sudo cp -r libmoveit_robot_trajectory.so.0.9.15 lib/
```
- To avoid error with installation of mpi4py package
```
sudo apt install libpython3.7-dev
pip3 install mpi4py
```
- No need to install mujoco-py, since the code uses Rviz
- Genetic algorithm library
```
pip3 install https://github.com/chovanecm/python-genetic-algorithm/archive/master.zip#egg=mchgenalg
https://github.com/adarshsehgal/python-genetic-algorithm.git
```


## How to run the program


**Before running roslaunch command, setup aubo robot repository with ros kinetic (link in pre requite section)**
```
cd catkin_workspace
catkin build
source devel/setup.bash
rosdep install openai_ros
```
**Training**

First clone the repository
```
git clone <github url> 
```
Training without simulation first, allows for the use of multiple CPU cores. 
To begin training using default values from openai (make sure to comment out lines with sys.exit() in train.py inorder for the python file to work as intended): 
```
cd ~/ga-drl-aubo

python3 train.py
```
The best policy will be generated in the directory `/tmp/newlog` (OR the log directory you specify).

To begin training using the GA_her+ddpg parameters generated from the python script `ga.py`
```
python3 train.py --polyak_value=0.924 --gamma_value=0.949 --q_learning=0.001 --pi_learning=0.001 --random_epsilon=0.584 --noise_epsilon=0.232
```

If you would like to recieve your own GA optimized parameters run (make sure to uncomment the line 89 and 94 in train.py inorder for ga.py to work as intended):
```
python3 ga.py
```
The best parameters will be saved to the file `bestParameters.txt`

**Launch rviz and moveit by either launch commands:** 

**For simulation:**
```
roslaunch aubo_i5_moveit_config demo.launch
```
For real robot:
```
roslaunch aubo_i5_robot_config moveit_planning_execution.launch robot_ip:=<your robot ip>
```
Go into the newher directory - `cd newher`

To run the connection between the robot gym environment and moveit run:
```
python2.7 moveit_motion_control.py
```

Run the genetic algorithm on her+ddpg while still in newher directory:
```
python3 ga.py
```

To make the program run faster, you can comment rviz below code in aubo robot repository:
```
cd ~/catkin_workspace/src/aubo_robot/aubo_i5_moveit_config/launch
gedit demo.launch (or just manually open demo.launch in any text editory)

 <!--<include file="$(find aubo_i5_moveit_config)/launch/moveit_rviz.launch">
    <arg name="config" value="true"/>
    <arg name="debug" value="$(arg debug)"/>
  </include> -->
```

Gym Environments available for GA-DRL execution (can be changes in ga.py):
- AuboReach-v0 - executes joint states with moveit
- AuboReach-v1 - only calculates actions but does not execute joint states (increased learning speed)
- AuboReach-v2 - only calculates actions but does not execute joint states (increased learning speed), reward fixes to converge the learning curve
- AuboReach-v3 - only calculates actions but does not execute joint states (increased learning speed), reward fixes to converge the learning curve, only 4 Aubo joints in action
- AuboReach-v4 - executes joint states with moveit, only run with first 4 joint states

## How to train the environment manually
```
python3 -m train --env=AuboReach-v2 --logdir=/tmp/openaiGA --n_epochs=70 --num_cpu=6 --polyak_value=0.184 --gamma_value=0.88 --q_learning=0.001 --pi_learning=0.001 --random_epsilon=0.055 --noise_epsilon=0.774
```
Values of all the above parameters can differ. These parameters are generated by GA-DRL algorithm, as mentioned in the paper, reference of which is mentioned in the start of this readme.

## How to play the environment with any chosen policy file
You can see the environment in action by providing the policy file:
```
python3 play.py <file_path>
python3 play.py /tmp/openaiGA/policy_best.pkl
```
where file_path = /tmp/openaiGA/policy_best.pkl in our case  .

## How to plot results:
For one set of parameter values and for one DRL run, plot results using:
```
python3 plot.py <dir>
python3 plot.py /tmp/openaiGA
```
where, dir = /tmp/openaiGA in our case, as mentioned in ga.py file. You can provide any log directory.
- If you are testing one set of parameters, in train.py, comment out the code which stops the system as soon as it reaches threshold success rate. 
```
sys.exit()
```

## Aubo environment setup details:
**NOTE**: Some experiments are using less than 6 joints for learning. Please refer to environments section for details.

**Action**: Each action is a set of 6 joint states of Aubo i5 robot

**Action space**: Each joint state can take joint values between -1.7 and 1.7

**Goal**: Random reachable point (a set of 6 joint states), [-0.503, 0.605, -1.676, 1.367, -1.527, -0.036]

**Initial State**: [0, 0, 0, 0, 0, 0]

**Reset state**: [0, 0, 0, 0, 0, 0]

**Reward**: Computed as absolute distance between corresponding joint states

**Success**: A joint state is considered success if distance between the joint states is less than 0.1

**Step**: 1. Take an action governing by DRL algorithm 2. compute reward 3. decide if it is success or not