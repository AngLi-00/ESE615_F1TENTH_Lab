# Lab 1: Intro to ROS 2

## Written Questions

### Q1: During this assignment, you've probably ran these two following commands at some point: ```source /opt/ros/foxy/setup.bash``` and ```source install/local_setup.bash```. Functionally what is the difference between the two?

Answer: 

We ran the first command to set up the main ROS2 environment since we need to source this script in every bash terminal we want to use ROS foxy in to have access to the ROS 2 commands.

The Second command mainly focus on your own workspace, which can add available packages from your own workspace to the environment (build the overlay based the main ROS2 environment). 

If you run the second command after running the first one, the packages in your own workspace (lab1_ws) will “replace” the packages of the underlying ROS2 in the current work environment.

### Q2: What does the ```queue_size``` argument control when creating a subscriber or a publisher? How does different ```queue_size``` affect how messages are handled?

Answer: 

The `queue_size` argument controls the size of the message queue used by a publisher or subscriber to buffer messages. 

When creating a publisher with a specific `queue_size`, you are specifying how many outgoing messages can be queued up if the rate of publishing exceeds the rate of transmission or processing.

For subscriber, if the queue size is set to a value greater than 1, incoming messages will be buffered in the subscriber's queue until they are processed by the callback function. This can help handle bursts of messages or brief spikes in message arrival without losing data.

Normally a larger queue size can help prevent message loss, but it may also increase memory usage. Conversely, a smaller queue size can reduce latency but may result in message loss if your system can't keep up with the message flow.

### Q3: Do you have to call ```colcon build``` again after you've changed a launch file in your package? (Hint: consider two cases: calling ```ros2 launch``` in the directory where the launch file is, and calling it when the launch file is installed with the package.)

Answer: 

If you are running the `ros2 launch` command from the directory where the launch file is located (the source directory of your package), you typically do not need to run `colcon build` again after modifying the launch file. ROS2 uses the launch file from the source directory directly. When you run `ros2 launch`, it will look for the launch files in the source directory of the package, and any changes you make to those launch files will be picked up without the need for a rebuild.

If you have previously built and installed your package using `colcon build`, and you are running `ros2 launch` with the installed package, you may need to rebuild the package to apply changes to launch files. This is because installed packages usually have their files (including launch files) copied to the install space during the build process.
