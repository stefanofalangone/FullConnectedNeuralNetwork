# -*- coding: utf-8 -*-
"""
Module that contains a loading from MNIST functions
==================================

Provides
   1. loadImages: returns a 28x28x[number of MNIST images] matrix containing the raw MNIST images
   2. loadLabes: returns an array of size [number of MNIST images] containing the images labels
"""

import struct as st
import numpy as np

def loadImages(filename):
    try:
        train_imagesfile = open(filename, "rb")

    except:
        print("You can't open this file. Please check if the filename is correct")

    train_imagesfile.seek(0)
    magic = st.unpack('>4B', train_imagesfile.read(4))

    nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
    nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column
    nImg = int(nImg/5.5)

    """
    print("magic: ",magic)
    print("nImg: ",nImg)
    print("nR: ",nR)
    print("nC: ",nC)
    """
    images_array = np.zeros((nImg,nR,nC))

    nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
    images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal)))
    images_array = images_array.reshape((nImg,nR,nC))
    np.transpose( images_array )

    train_imagesfile.close()

    return images_array

def loadLabels(filename):
    try:
        train_labelsfile = open(filename, "rb")

    except:
        print("You can't open this file. Please check if the filename is correct")

    train_labelsfile.seek(0)
    magic = st.unpack('>4B', train_labelsfile.read(4))

    nItm = st.unpack('>I',train_labelsfile.read(4))[0] #num of items
    """
    print("magic: ",magic)
    print("nItm: ",nItm)
    """
    labels_array = np.zeros(nItm)

    nBytesTotal = nItm*1 #since each pixel data is 1 byte
    labels_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_labelsfile.read(nBytesTotal)))

    train_labelsfile.close()

    return labels_array