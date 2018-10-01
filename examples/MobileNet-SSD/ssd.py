import numpy as np
import sys.os
import cv2
caffe_root = '/data/home/jeffhzhang/caffe-ssd'
sys.path.insert(0, caffe_root + 'python')
import caffe
import time

CLASSES = ('background', 'person', 'background')

class SSD():
    N=1;C=3;H=300;W=300
    def __init__(self, gpu_id=0, proto_name = '/data/home/jeffhzhang/caffe-ssd/examples/Mobile-SSD/example/MobileSSD_deploy_dw.prototxt', model_name = '/data/home/jeffhzhang/caffe-ssd/examples/Mobile-SSD/snapshot/MobileSSD_deploy.caffemodel'):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        self.net = caffe.Net(proto_name, model_name, caffe.TEST)

    def get_shape(self):
        SSD.N, SSD.C, SSD.H, SSD.W = self.net.blobs['data'].data.shape
        return SSD.C, SSD.H, SSD.W

    def detect(self, image_data):
        caffe.set_mode_gpu()
        image_num = len(image_data)
        self.net.blobs['data'].reshape(image_num, SSD.C, SSD.H, SSD.W)
        self.net.blobs['data'].data[:] = image_data
        t = time.time()
        out = self.net.forward()
        print "time cost of forward is: " + str(time.time() - t) + "s."
        box = out['detection_out'][0,0,:,3:7]
        cls = out['detection_out'][0,0,:,1]
        conf = out['detection_out'][0,0,:,2]

        res_list = []
        for i in range(len(box)):
            if int(cls[i]) == 1:
                res = [conf[i], box[i][0], box[i][1], box[i][2], box[i][3]]
                res_list.append(res)

        return res_list

