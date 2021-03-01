#!/bin/bash
docker run -p 4567:4567 -v $PWD:/capstone -it ubuntu:capstone_image /bin/bash -- ./start.sh
