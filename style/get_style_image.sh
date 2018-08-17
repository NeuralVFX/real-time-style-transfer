#!/bin/bash

URL=https://upload.wikimedia.org/wikipedia/en/7/71/Francis_Picabia%2C_1912%2C_Tarentelle%2C_oil_on_canvas%2C_73.6_x_92.1_cm%2C_Museum_of_Modern_Art%2C_New_York.jpg
IMG_FILE=./style/franc.jpg
wget $URL -O $IMG_FILE

URL=https://upload.wikimedia.org/wikipedia/commons/3/36/Vassily_Kandinsky%2C_1912_-_Improvisation_27%2C_Garden_of_Love_II.jpg
IMG_FILE=./style/vassily.jpg
wget $URL -O $IMG_FILE

URL=https://cdn.skim.gs/image/upload/v1456338926/msi/picasso-mediterranean-landscape_fdmvxb.jpg
IMG_FILE=./style/picasso.jpg
wget $URL -O $IMG_FILE
