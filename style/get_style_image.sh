#!/bin/bash

URL=https://upload.wikimedia.org/wikipedia/en/7/71/Francis_Picabia%2C_1912%2C_Tarentelle%2C_oil_on_canvas%2C_73.6_x_92.1_cm%2C_Museum_of_Modern_Art%2C_New_York.jpg
IMG_FILE=./data/franc.jpg
wget $URL -O $IMG_FILE
