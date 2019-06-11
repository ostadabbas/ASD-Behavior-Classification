import argparse
import sys
import os
import glob
import cv2 as cv
import subprocess
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Run all videos')
    parser.add_argument(
        '--s', '-s', dest='s', required=True,
        help='set1/set2/set3')
    return parser.parse_args()


def _run_cmd(tool_file,s):
    if s=='set3' or s=='set1' or s=='set4':
      print('Processing set1/set3/set4 Videos......')
      csv_file = '../../hddDrive2/datasets/DetectAndTrack_pose_output/phase3/'+s+'.csv'
      df =pd.read_csv(csv_file)
      df.dsm_category.replace('A. Persistent deficits in social communication and social interaction across multiple contexts, as manifested by the following, currently or by history:', 1, regex=True, inplace=True)
      df.dsm_category.replace('B. Restricted, repetitive patterns of behavior, interests, or activities, as manifested by at least two of the following, currently or by history:', 1, regex=True, inplace=True)
      df.dsm_category.replace('Typical', 0, regex=True, inplace=True)
      df.dsm_category.replace('Associated Feature', 1, regex=True, inplace=True)
      label = df.dsm_category.values
      ts = df.seconds_offset.values
      stamps = np.zeros(len(ts))
      for i,t in enumerate(ts):
        stamps[i] = int(np.round(t*30))
      
      videoname = df.real_filename.values
      videos = []
      for v in videoname:
        fullname = '/datasets/'+s+'_videos/vids/' + v
        videos.append(fullname)
      
      all_data = set()
      for i in range(len(label)):
        all_data.add((videos[i], ts[i], label[i]))
      
      
      all_data=np.array(list(all_data))
      videos = list(all_data[:,0])
      ts = list(all_data[:,1])
      label = list(all_data[:,2])
      
      seen=[]
      f=open('/hddDrive2/datasets/DetectAndTrack_pose_output/phase1/potion/train_label.txt','a+')
      final_data = []
      for i in range(len(videos)):
        if os.path.exists(videos[i]):
          cap = cv.VideoCapture(videos[0])
          cap.set(1, stamps[0])
          ret, frame = cap.read()
          if frame is not None:
            v = videos[i]
            o = '/hddDrive2/datasets/DetectAndTrack_pose_output/phase1/'+s+'_video_results/'+v.split('/')[-1][:-4]
            t = ts[i]
            original_size = frame.shape[0:2]
            if (v.split('/')[-1],float(t)) not in seen:
              seen.append((v.split('/')[-1],float(t)))
              
              final_data.append((o+'/'+o.split('/')[-1]+'_'+str(t)+'_detections.pkl', original_size, int(label[i])))
              
              print('Current Video is %s at time %s'%(v,t))
              cmd = '''python tools/{tool_file} \
                 --video {v} \
                 --output {o} \
                 --t {t}
                  '''.format(tool_file=tool_file,v=v,o=o,t=float(t))
              subprocess.call(cmd, shell=True)  
            else:
              final_data.remove((o+'/'+o.split('/')[-1]+'_'+str(t)+'_detections.pkl', original_size, 1-int(label[i])))
              final_data.append((o+'/'+o.split('/')[-1]+'_'+str(t)+'_detections.pkl', original_size, 0))
          
      for d in final_data:
        print >>f,d[0]+' '+str(d[1])+' '+str(d[2])
      f.close()
            
    
    elif s=='set2':
      print('Processing set2 Videos......')
      df =pd.read_csv('../../hddDrive2/datasets/DetectAndTrack_pose_output/phase3/set2.csv')
      df.dsm_category.replace('A. Persistent deficits in social communication and social interaction across multiple contexts, as manifested by the following, currently or by history:', 1, regex=True, inplace=True)
      df.dsm_category.replace('B. Restricted, repetitive patterns of behavior, interests, or activities, as manifested by at least two of the following, currently or by history:', 1, regex=True, inplace=True)
      df.dsm_category.replace('Typical', 0, regex=True, inplace=True)
      df.dsm_category.replace('Associated Feature', 1, regex=True, inplace=True)
      label = df.dsm_category.values
      ts = df.seconds_offset.values
      stamps = np.zeros(len(ts))
      for i,t in enumerate(ts):
        stamps[i] = int(np.round(t*30))
      
      videoname = df.real_filename.values
      videos = []
      for v in videoname:
        fullname = '/datasets/set2_videos/new_videos/' + v
        videos.append(fullname)
      
      all_data = set()
      for i in range(len(label)):
        all_data.add((videos[i], ts[i], label[i]))
      
      all_data=np.array(list(all_data))
      videos = list(all_data[:,0])
      ts = list(all_data[:,1])
      label = list(all_data[:,2])
      
      seen=[]
      f=open('/hddDrive2/datasets/DetectAndTrack_pose_output/phase4/potion/test_label.txt','w')
      final_data = []
      for i in range(len(videos)):
        if os.path.exists(videos[i]):
          cap = cv.VideoCapture(videos[0])
          cap.set(1, stamps[0])
          ret, frame = cap.read()
          if frame is not None:
            v = videos[i]
            o = '/hddDrive2/datasets/DetectAndTrack_pose_output/phase4/'+s+'_video_results/'+v.split('/')[-1][:-4]
            t = ts[i]
            original_size = frame.shape[0:2]
            if (v.split('/')[-1],float(t)) not in seen:
              seen.append((v.split('/')[-1],float(t)))
              
              final_data.append((o+'/'+o.split('/')[-1]+'_'+str(t)+'_detections.pkl', original_size, int(label[i])))
              
              print('Current Video is %s at time %s'%(v,t))
              cmd = '''python tools/{tool_file} \
                 --video {v} \
                 --output {o} \
                 --t {t}
                  '''.format(tool_file=tool_file,v=v,o=o,t=float(t))
              subprocess.call(cmd, shell=True)  
            else:
              final_data.remove((o+'/'+o.split('/')[-1]+'_'+str(t)+'_detections.pkl', original_size, 1-int(label[i])))
              final_data.append((o+'/'+o.split('/')[-1]+'_'+str(t)+'_detections.pkl', original_size, 0))
          
      for d in final_data:
        print >>f,d[0]+' '+str(d[1])+' '+str(d[2])
      
    
def main():
    args = parse_args()
    tool_file = 'test_on_single_video.py'
    _run_cmd(tool_file, args.s)


if __name__ == '__main__':
    main()
