## Setup for ubuntu
1. Install python3
- `sudo nano /etc/apt/sources.list`
    deb http://archive.canonical.com/ubuntu xenial partner
    deb-src http://archive.canonical.com/ubuntu xenial partner
    deb http://archive.ubuntu.com/ubuntu xenial universe main restricted multiverse
    deb http://security.ubuntu.com/ubuntu/ xenial-security main universe multiverse restricted
    deb http://archive.ubuntu.com/ubuntu xenial-updates main universe multiverse restricted
    deb http://archive.ubuntu.com/ubuntu xenial-backports main universe multiverse restricted
- `sudo apt-get update`
- `sudo apt-get install -y python3-pip`

(optional)
- `export LC_ALL="en_US.UTF-8"`
- `export LC_CTYPE="en_US.UTF-8"`
- `sudo dpkg-reconfigure locales`

- `pip3 install pyinstaller`
- build by running `bash build.sh`
- put the .tar.xz file to /tmp
- run install.sh

## Features
- Visit localhost:5000/video_feed, it will return face, age and gender in real-time
- 2 RPC call available
  - START_DETECTING_FACE : start capturing faces and store them in array
  - END_DETECTING_FACE : end capturing faces and take the most accurate photo (mode of gender, mean of age, photo with age closest to its mean) then return the photo buffer

## Parameters

### Camera related
- LIVELINESS: true then detect eye blinking, age and gender
- AGE_WEIGHTED_AVERAGE: true then take weighted average, else take the argmax
- TEST_MODE: true if run locally
- SHOW_URL: true if run locally
- SAVE_PHOTO: true then save photo
- DETECT_TIMEOUT: value for detection timeout
- ROTATION: 0 = 90 clockwise, 1 = 180, 2 = 90 anti-clockwise

### RabbitMQ related
- RABBITMQ_USERNAME : username of rabbitmq
- RABBITMQ_PASSWORD : password of rabbitmq

## Reference from:
- [Gaze Tracking](https://github.com/antoinelame/GazeTracking)
- [Age and Gender Caffe Model](https://gist.github.com/GilLevi/c9e99062283c719c03de)