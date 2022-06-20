#-*- coding: utf-8 -*-
"""
@author: Md. Rezwanul Haque

"""
import os 
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def splitVideo(args):
    required_video_file = args.video_data 
    if not os.path.exists(os.path.dirname(args.split_videos)):
        os.mkdir(os.path.dirname(args.split_videos))
    with open(args.times) as f:
        times = f.readlines()
        times = [x.strip() for x in times] 
    for time in times:
        starttime = int(time.split("-")[0])
        endtime = int(time.split("-")[1])
        ffmpeg_extract_subclip(required_video_file, starttime, endtime,\
                targetname= os.path.join(args.split_videos, "part_"+str(times.index(time)+1)+"_"+str(required_video_file.split("/")[-1])) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Code to create video splits with required time line")
    parser.add_argument("-s", "--video_data", type=str, required=True, help="path to the video file name (required)")
    parser.add_argument("-d", "--split_videos", type=str, required=True, help="path to the destination folder where split video file will store (required)")
    parser.add_argument("-t", "--times", type=str, default="times.txt", required=True, help="path to the times.txt file name (default: 'times.txt')")

    args = parser.parse_args()

    splitVideo(args)

    '''python

    $ python split_video.py -s anatomy_videos/General_embryology_Part-1.mp4 -d split_videos/ -t times.txt

    ''' 
