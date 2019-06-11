import argparse
import sys
import os
import subprocess


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
    else:
        tool_file = 'prediction.py'
        testing(tool_file)


if __name__ == '__main__':
    mode='train'
    main(mode)