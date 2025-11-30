#!/bin/bash
FILE=$1

if [[ $FILE != "cityscapes" &&  $FILE != "night2day" &&  \
      $FILE != "edges2handbags" && $FILE != "edges2shoes" &&  \
      $FILE != "facades" && $FILE != "maps" ]]; then
  echo "Available datasets are cityscapes, night2day, edges2handbags, edges2shoes, facades, maps"
  exit 1
fi

echo "Specified [$FILE]"

URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./assets/$FILE.tar.gz
TARGET_DIR=./assets/$FILE/

mkdir -p $TARGET_DIR
wget -N $URL -O $TAR_FILE
tar -zxvf $TAR_FILE -C ./assets/
rm $TAR_FILE