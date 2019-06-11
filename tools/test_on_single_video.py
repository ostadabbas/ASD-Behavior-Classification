#Script to run DetectandTrack Code on a single video which may or may not belong to any dataset.
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
os.environ['GLOG_minloglevel'] = '2' 
import os.path as osp
import sys
sys.path.append('/localDrive/DetectAndTrack/lib')
import numpy as np
import pickle
import cv2
import argparse
import shutil
import yaml
import glob
import time
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test_engine import initialize_model_from_cfg, empty_results, extend_results
from core.test import im_detect_all
from utils.io import robust_pickle_dump
import utils.c2



def parse_args():
    parser = argparse.ArgumentParser(description='Run DetectandTrack on a single video and visualize the results')
    parser.add_argument(
        '--cfg', '-c', dest='cfg_file', default='configs/video/2d_best/01_R101_best_hungarian.yaml',
        help='Config file to run')
    parser.add_argument(
        '--video', '-v', dest='video_path',
        help='Path to Video',
        required=True)
    parser.add_argument(
        '--t', '-t', dest='t',
        help='Autism Time Stamp',
        required=True)
    parser.add_argument(
        '--output', '-o', dest='out_path',
        help='Path to Output')
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def _read_video(args):
    vidcap = cv2.VideoCapture(args.video_path)
    success,image = vidcap.read()
    count = 1
    count2 = 1
    success = True
    t = float(args.t)
    temp_frame_folder = osp.join(args.out_path,args.vid_name + '_frames/',str(t))
    start = int(np.round(t*30))-240
    if start<=0:
      start=1
    end = int(np.round(t*30))+60
    if os.path.exists(temp_frame_folder):
      shutil.rmtree(temp_frame_folder)
    os.makedirs(temp_frame_folder)
    while success:
        if count>=start and count<=end:
          cv2.imwrite(osp.join(temp_frame_folder,'%08d.jpg' % count2), image)     # save frame as JPEG file
          count2 += 1
        success,image = vidcap.read()
        count +=1
    return count2-1


def main(name_scope, gpu_dev, num_images, args):
    t=args.t
    model = initialize_model_from_cfg()
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    temp_frame_folder = osp.join(args.out_path,args.vid_name + '_frames/',str(t))
    imgs = glob.glob(temp_frame_folder+'/*.jpg')
    for i in range(len(imgs)):
        if i%100==0:
          print('Processing Detection for Frame %d'%(i+1))
        im_ = cv2.imread(imgs[i])
        assert im_ is not None
        im_ = np.expand_dims(im_, 0)
        with core.NameScope(name_scope):
            with core.DeviceScope(gpu_dev):
                cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                    model, im_, None)                                        #TODO: Parallelize detection

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

    det_name = args.vid_name + '_' + str(args.t) + '_detections.pkl'
    det_file = osp.join(args.out_path, det_name)
    robust_pickle_dump(dict(all_keyps=all_keyps),det_file)
    shutil.rmtree(osp.join(args.out_path, args.vid_name + '_frames'))


if __name__=='__main__':
    os.environ['GLOG_minloglevel'] = '2' 
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    if args.out_path == None:
        args.out_path = args.video_path
    args.vid_name = args.video_path.split('/')[-1].split('.')[0]

    utils.c2.import_custom_ops()
    utils.c2.import_detectron_ops()
    utils.c2.import_contrib_ops()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_and_infer_cfg()
    num_images = _read_video(args)
    gpu_dev = core.DeviceOption(caffe2_pb2.CUDA, cfg.ROOT_GPU_ID)
    name_scope = 'gpu_{}'.format(cfg.ROOT_GPU_ID)
    main(name_scope, gpu_dev, num_images, args)