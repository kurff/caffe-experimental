import numpy as np  
import sys,os  
import cv2
caffe_root = '/data/home/jeffhzhang/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time

caffe.set_device(1)
caffe.set_mode_gpu()
net_file= 'example/MobileNetSSD_deploy_dw.prototxt'  
caffe_model='snapshot/MobileNetSSD_deploy.caffemodel'  
#net_file= '/data/home/jeffhzhang/MobileNet-SSD-master/example/MobileNetSSD_deploy.prototxt'
#caffe_model= '/data/home/jeffhzhang/MobileNet-SSD-master/snapshot/MobileNetSSD_deploy.caffemodel'
test_dir = "images"
#test_dir = "img"
#test_dir = "/data1/jeffhzhang/video-data/test1"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    t = time.time()
    net.blobs['data'].data[...] = img
    #t = time.time()
    out = net.forward()  
    #print "time cost is: " + str(time.time() - t) + "s."
    box, conf, cls = postprocess(origimg, out)
    print "time cost is: " + str(time.time() -t) + "s."

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    #print imgfile
    #cv2.imwrite('predictions/' + imgfile, origimg)
    cv2.imwrite(imgfile, origimg)
 
#    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
#    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
