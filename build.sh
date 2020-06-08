#!/bin/bash
VERSION="0.0.0"

if [ $# == 0 ]; then
    echo -n 'Version: ' 
    read VERSION
else
	VERSION=$1
fi

if [[ -d build ]]; then
    rm -R build
fi

pyinstaller --onefile --noupx server.py
cp -r data ./dist/
cp -r lib ./dist/
cp -r haarcascade_frontalface_alt.xml ./dist/
cp -r gaze_tracking/trained_models ./dist/
cp -r start.sh ./dist/
tar -vcf face-detect_$VERSION.tar.xz dist


