#!/bin/bash

URL=http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
TAR_FILE=./data/VOCtrainval_06-Nov-2007.tar
wget $URL -O $TAR_FILE
tar -xvzf $TAR_FILE -C ./data/