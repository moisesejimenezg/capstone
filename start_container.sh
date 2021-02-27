#!/bin/bash
docker run -p 4567:4567 -v /home/mjimenez/development/udacity/car_nanodegree/final_project:/capstone -it ubuntu:capstone_image /bin/bash -- ./start.sh
