### Run Franka Panda with MoveIt 

To start Franka Panda with MoveIt, first make sure that it has the blue light on its base. If not, make sure that the emergency switch is not pressed. 

Then open a terminal and run the following commands 


```bash
$ cd 
$ cd franka_ws/
$ source devel/setup.bash
$ roslaunch franka_lcas_experiments franka_moveit.launch
```


### Run The Python Node 

To test your code, please do the following: 

```bash
# open another terminal (make sure the robot with moveit is on) 

$ cd 
$ cd franka_ws/ 
$ source devel/setup.bash

# rosrun <package-name> <node-name>
$ rosrun franka_lcas_experiments advance_robotics_week_8.py
```

If you create a new python script, make sure that it is executable by giving it executable permissions as follow: 

```bash
# open a terminal and cd to the directory where the python node file is. 
$ sudo chmod +x python-file-name.py 
```

*Note: In case of any unwanted movement from the robot, please press the emergency switch button as soon as possible.*