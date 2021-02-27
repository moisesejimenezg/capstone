#!/bin/bash

# install dependencies
apt update
apt install curl -y
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
apt update
apt install git -y
apt install gcc -y
apt install -y python python-dev
apt install ros-kinetic-desktop-full -y

curl -O https://bootstrap.pypa.io/2.7/get-pip.py
python get-pip.py
rm get-pip.py
python -m pip install --upgrade "pip < 21.0"
