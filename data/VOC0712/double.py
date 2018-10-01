import os
import numpy as np

filelist = open('person_train.txt','r')
filelist2 = open('person_train2.txt','w')

line = filelist.readline()
while line:
    line = line.strip()
    label = line[:line.find('JPEGImages')] + 'Annotations' + line[line.find('JPEGImages') + 10:-3] + 'xml'
#    print label
    filelist2.write(line + ' ' + label + '\n')
    line = filelist.readline()

filelist.close()
filelist2.close()
