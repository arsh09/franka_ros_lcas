### Data for Deep ProMP 

This data was collected on 11th August. This readme is for data_51_samples folder.

### Collection Method: 

Breast phantom was placed on different locations, making sure that it was in realsense-camera's field of view. Then based on 2D to 3D point projection, the 3d point was found for the equivalent 2d point on color image for nipple. This 3D point was transformed into base frame from camera optical frame. Then MoveIt was used to make the arm go to this transformation. Please note that I had to change the URDF and moveit packages for the franka panda as because moveit for generic panda package plans for panda_link8 (with planning frame being panda_link0) but because we have xela_sensor (which actually make contact with the phantom), I changed the arm-group definition a little bit to include this xela_sensor_frame (for each of use to be honest). 

### Data Structure: 

I recorded 50 samples of ROS bag file like this: 

1) Robot is at home pose (the same home pose we used for Yetty experiments) 
2) Start data recording (recording joint values, point cloud, color, depth images and tf frames etc).
3) Robot moves and xela sensor makes contact with the nipple on the phatom
4) As soon the contact is made, data recording is stopped
5) The robot goes to home pose again and repeat the same thing with different 3D location of the phantom 


### Preprocessing: 

I pre-processed the rosbag files so that you have data properly in 4 different files (for each experiment or a different localtion). These are: 

``` 
   - data_51_samples/experiment_X.json  // contains trajectories and other data
   - data_51_samples/experiment_X_color.npy // color image (from home pose)
   - data_51_samples/experiment_X_depth.npy // depth image (from home pose)
   - data_51_samples/experiment_X_pointcloud.npy // point cloud XYZRGB (from home pose)
```

### Note: 

- Color images are in numpy array format. 
- Depth images are in numpy array format. Visualizing them on matlplotlib might not look nice as the depth is in 11-bit format (not exactly gray i.e. 8-bit, or RGB i.e. 24-bit). 
- Point cloud is alsi in numpy array with this size (X, 6) where 6-> x y z r g b
- JSON file contains joint-space trajectory and its equivalent cartesian space trajectory (from panda_link0 to xela_sensor_frame). It also has joint torques , speed, time etc. 


### Experiment start/stop location: 

The following image shows the start and stop XY in world coordinates. The end location can be think of nipple position (as the xela sensor frame makes contact with it at the end of trajectory). The start position is basically the home pose. 


