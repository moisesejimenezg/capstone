#!/bin/bash
catkin_make
source devel/setup.sh
roslaunch src/twist_controller/launch/dbw_test.launch
