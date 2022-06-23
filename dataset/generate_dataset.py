#-*- coding: utf-8 -*-
"""
@author: Md. Rezwanul Haque

"""
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os
from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py
import glob
import shutil
import matplotlib.pyplot as plt
from KTS.cpd_auto import cpd_auto
from KTS.utils import LOG_INFO, create_dir
import sys
sys.path.append('..')
from models.CNN import ResNet


class Generate_Dataset:
    """
        Generate Dataset:
            1. Converting video to frames
            2. Extracting features
            3. Getting change points
            4. User Summary ( for evaluation )

    """
    def __init__(self, video_path, save_path, frame_dir, train_data):
        self.resnet = ResNet()
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.h5_file = h5py.File(save_path, 'w') 
        self.frame_root_path = frame_dir
        self.train_data = train_data

        self._set_video_list(video_path)

    def _set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = os.listdir(video_path)
            self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            LOG_INFO("LIST OF VIDEOS IS BELLOW: ")
            LOG_INFO(file_name)
            self.dataset['video_{}'.format(idx+1)] = {}
            self.h5_file.create_group('video_{}'.format(idx+1))


    def _extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        res_pool5 = self.resnet(frame)
        frame_feat = res_pool5.cpu().data.numpy().flatten()

        return frame_feat

    def _get_change_points(self, video_feat, n_frame, fps, plot_fig=False):
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _scores = cpd_auto(K, m, 1)

        if plot_fig:
            plt.ioff()
            # ===============================================================
            plt.figure("Test 4: Frames: automatic selection of the number of change-points")
            # (X, cps_gt) = gen_data(n, m)
            # print ("Ground truth: (m=%d)" % m, cps_gt)

            plt.plot(video_feat)
            K = np.dot(video_feat, video_feat.T)
            cps = change_points
            scores = _scores
            print("Estimated: (m=%d)" % len(cps), cps)
            print('scores: {}  lenght: {}'.format(scores, len(scores)))
            mi = np.min(video_feat)
            ma = np.max(video_feat)
            for cp in cps:
                plt.plot([cp, cp], [mi, ma], 'r')
            plt.show()
            print ("="*79)
        
        ###
        change_points = np.concatenate(([0], change_points, [n_frame-1]))
        
        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = 1 # we choose only one frame
            temp_n_frame_per_seg.append(n_frame)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

        # LOG_INFO('CPS: {}'.format(change_points))
        # LOG_INFO('LEN CPS: {}'.format(len(change_points)))
        # LOG_INFO('N_FRAMES: {}'.format(n_frame_per_seg))
        
        return change_points, n_frame_per_seg


    # TODO : save dataset
    def _save_dataset(self, video_idx, video_feat_for_train, picks, n_frames, fps, video_filename, change_points, n_frame_per_seg):
        self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)
        self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
        self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
        self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps
        self.h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_filename.split('.')[0]
        self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points
        self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg

    def generate_dataset(self, plot_fig=False):
        LOG_INFO('[INFO] CNN processing...')
        for video_idx, video_filename in enumerate(self.video_list):
            video_path = video_filename
            if ".h5" in video_path:
                continue
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            LOG_INFO('VIDEO PATH: {}'.format(video_path))

            _video_path = video_path.split("/")[-1]
            video_basename = os.path.basename(_video_path).split('.')[0]

            video_capture = cv2.VideoCapture(video_path)

            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_list = []
            picks = []
            video_feat = None
            video_feat_for_train = None
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = video_capture.read()
                if success:
                    frame_feat = self._extract_feature(frame)

                    if frame_idx % 15 == 0:
                        picks.append(frame_idx)

                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))

                    if video_feat is None:
                        video_feat = frame_feat
                    else:
                        video_feat = np.vstack((video_feat, frame_feat))

                    if self.train_data:
                        # this is for frames extraction
                        img_filename = "{}.jpg".format(str(frame_idx).zfill(6))
                        train_data_frames_dir = create_dir(self.frame_root_path, video_basename)

                        if not os.path.exists(os.path.join(train_data_frames_dir, img_filename)):
                            cv2.imwrite(os.path.join(train_data_frames_dir, img_filename), frame)
                else:
                    break

            video_capture.release()

            change_points, n_frame_per_seg = self._get_change_points(video_feat, n_frames, fps, plot_fig)
            
            if self.train_data:
                ### Seperate the selected frames after applying KTS algo.
                train_data_frames_dir = create_dir(self.frame_root_path, video_basename)
                LOG_INFO('TR DATA FRM: {}'.format(train_data_frames_dir))
                
                src_dir = train_data_frames_dir
                ext = "separate_"+train_data_frames_dir.split("/")[-1]
                dest_path=create_dir(train_data_frames_dir,ext)
                LOG_INFO('SEP TR DATA FRM: {}'.format(dest_path))
                
                for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
                    _file_name = jpgfile.split("/")[-1]
                    file_num = _file_name.split(".")[0]
        
                    if int(file_num) in change_points:
                        shutil.copy2(jpgfile, dest_path)

            ## Save Dataset
            self._save_dataset(video_idx, video_feat_for_train, picks, n_frames, fps, video_filename, change_points, n_frame_per_seg)
