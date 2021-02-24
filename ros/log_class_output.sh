#!/bin/bash
rostopic echo /rosout | grep "$1"
