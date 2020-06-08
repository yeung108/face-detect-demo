#!/bin/bash

SOURCELIST='/etc/apt/sources.list'

sudo grep -qxF 'deb http://archive.canonical.com/ubuntu xenial partner' $SOURCELIST || echo 'deb http://archive.canonical.com/ubuntu xenial partner' >> $SOURCELIST
sudo grep -qxF 'deb-src http://archive.canonical.com/ubuntu xenial partner' $SOURCELIST || echo 'deb-src http://archive.canonical.com/ubuntu xenial partner' >> $SOURCELIST
sudo grep -qxF 'deb http://archive.ubuntu.com/ubuntu xenial universe main restricted multiverse' $SOURCELIST || echo 'deb http://archive.ubuntu.com/ubuntu xenial universe main restricted multiverse' >> $SOURCELIST
sudo grep -qxF 'deb http://security.ubuntu.com/ubuntu/ xenial-security main universe multiverse restricted' $SOURCELIST || echo 'deb http://security.ubuntu.com/ubuntu/ xenial-security main universe multiverse restricted' >> $SOURCELIST
sudo grep -qxF 'deb http://archive.ubuntu.com/ubuntu xenial-updates main universe multiverse restricted' $SOURCELIST || echo 'deb http://archive.ubuntu.com/ubuntu xenial-updates main universe multiverse restricted' >> $SOURCELIST
sudo grep -qxF 'deb http://archive.ubuntu.com/ubuntu xenial-backports main universe multiverse restricted' $SOURCELIST || echo 'deb http://archive.ubuntu.com/ubuntu xenial-backports main universe multiverse restricted' >> $SOURCELIST

sudo apt-get update -y
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-venv

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libx11-dev libatlas-base-dev
sudo apt-get install libgtk-3-dev libboost-python-dev

tar -xvf /tmp/face-detect.tar.xz -C /opt
cd /opt/face-detect
sudo chmod 777 start.sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

sudo cp face-detect.service /etc/systemd/system/
sudo systemctl start face-detect.service
sudo systemctl daemon-reload

