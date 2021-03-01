#!/bin/bash

docker pull ubuntu:xenial
docker build -t "xenial:capstone" .
