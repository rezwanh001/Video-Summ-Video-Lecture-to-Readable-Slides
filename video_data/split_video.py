#-*- coding: utf-8 -*-
"""
@author: Md. Rezwanul Haque

"""
import os 
import argparse
from glob import glob
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def splitVideo(args):
    """
        Split long video into small chunks
        args:
            video_data  :       source path of video file (.mp4) 
            split_videos:       destination path of processed chunk video files (.mp4)
            times       :       path of times.txt file which is included 'start-time' and 'end-time' in seconds.
        returns:
            splited videos into the destination folder 
        
    """
    # path of long video data for spliting 
    required_video_file = args.video_data 
    # Add file formats here
    ext = ['mp4', 'avi', 'mov', 'wmv']    
    video_files = []
    [video_files.extend(glob(required_video_file + '*.' + e)) for e in ext]

    # check and create directory for storing split video files
    if not os.path.exists(os.path.dirname(args.split_videos)):
        os.mkdir(os.path.dirname(args.split_videos))

    # videos files access
    for vid_file in tqdm(video_files):
        # for accessing time text file
        time_txt = (vid_file.split("/")[-1]).split(".")[0]
        time_txt_path = os.path.join(required_video_file, time_txt+".txt")

        # read text files
        with open(time_txt_path) as f:
            times = f.readlines()
            times = [x.strip() for x in times] 
        # split video according to the time line (in seconds)
        for time in times:
            starttime = int(time.split("-")[0])
            endtime = int(time.split("-")[1])
            ffmpeg_extract_subclip(vid_file, starttime, endtime,\
                    targetname= os.path.join(args.split_videos, "part_"+str(times.index(time)+1)+"_"+str(vid_file.split("/")[-1])) )

if __name__ == "__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Code to create video splits with required time line")
    parser.add_argument("-s", "--video_data", type=str, default='anatomy_videos', help="path to the video file name (required)")
    parser.add_argument("-d", "--split_videos", type=str, default='split_videos', help="path to the destination folder where split video file will store (required)")

    args = parser.parse_args()

    splitVideo(args)


    '''python
    $ python split_video.py -s anatomy_videos/ -d split_videos/
    ''' 
