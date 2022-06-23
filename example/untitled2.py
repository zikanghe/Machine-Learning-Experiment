# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:34:27 2022

@author: Zikang He
"""

import numpy as np

import dapper.mods as modelling
import time
from multiprocessing import freeze_support, get_start_method
# Importing the model's modules


xx = np.load('true2.npy')
xx1 = np.ones((40000,36))
xx1[:,0:10]=xx[10001:50001,0:10]
xx1[:,10:20]=xx[10001:50001,20:30]
xx1[:,20:35]=xx[10001:50001,40:55]
xx1[:,35]=xx[10001:50001,55]
xx=xx1
f = open('dts1_F.dat')
f1=np.loadtxt(f)
f1 = f1[0:40000]
