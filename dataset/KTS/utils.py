#-*- coding: utf-8 -*-
"""
@author: MD. Nazmudduha Ansary
@adaptation: Md. Rezwanul Haque
"""

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os

#---------------------------------------------------------------
# common utils
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------