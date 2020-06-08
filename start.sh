#!/bin/bash

# INPUT env variable CAMTYPE 'rs' for RealSense, 'opencv' for other cameras

# ROTATION: 0 = 90 clockwise, 1 = 180, 2 = 90 anti-clockwise

LD_LIBRARY_PATH='lib'
PORT='5000'
LIVELINESS=false
AGE_WEIGHTED_AVERAGE=true
TEST_MODE=false
SHOW_URL=false
SAVE_PHOTO=false
DETECT_TIMEOUT=50.0
ROTATION=-1
RABBITMQ_USERNAME=your_username
RABBITMQ_PASSWORD=your_password
#source venv/bin/activate

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib PORT=$PORT RABBITMQ_USERNAME=$RABBITMQ_USERNAME RABBITMQ_PASSWORD=$RABBITMQ_PASSWORD LIVELINESS=$LIVELINESS AGE_WEIGHTED_AVERAGE=$AGE_WEIGHTED_AVERAGE TEST_MODE=$TEST_MODE DETECT_TIMEOUT=$DETECT_TIMEOUT SHOW_URL=$SHOW_URL SAVE_PHOTO=$SAVE_PHOTO ROTATION=$ROTATION ./server
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib PORT=$PORT RABBITMQ_USERNAME=$RABBITMQ_USERNAME RABBITMQ_PASSWORD=$RABBITMQ_PASSWORD LIVELINESS=$LIVELINESS AGE_WEIGHTED_AVERAGE=$AGE_WEIGHTED_AVERAGE TEST_MODE=$TEST_MODE DETECT_TIMEOUT=$DETECT_TIMEOUT SHOW_URL=$SHOW_URL SAVE_PHOTO=$SAVE_PHOTO ROTATION=$ROTATION python3 server.py