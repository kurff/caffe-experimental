import os
import numpy as np
import random

filelist = open('person_train_4k.txt','r')
filelist2 = open('person_train_4k_shuffle.txt','w')

lines = filelist.readlines()

random.shuffle(lines)

for line in lines:
    filelist2.write(line)

filelist.close()
filelist2.close()
