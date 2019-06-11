import argparse
import sys
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Classification Network')
    parser.add_argument(
        '--m', '-m', dest='m', required=True,
        help='train or test')
    return parser.parse_args()
    
def training(tool_file):
    for f in range(1,6):
        cmd = '''python {tool_file} --f {f}'''.format(tool_file=tool_file,f=f)
        subprocess.call(cmd, shell=True) 

def testing(tool_file):
    for e in range(1,6):
        cmd = '''python {tool_file} --e {e}'''.format(tool_file=tool_file,e=e)
        subprocess.call(cmd, shell=True)
  
def main(mode):
    if mode=='train':
        tool_file = 'model_cv.py'
        training(tool_file)
    elif mode=='test':
        tool_file = 'prediction.py'
        testing(tool_file)
    else:
        print('You have to enter train or test')

if __name__ == '__main__':
    args = parse_args()
    mode = args.m
    main(mode)