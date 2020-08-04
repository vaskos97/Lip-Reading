import os
import numpy as np
import cv2
from tqdm import tqdm
#from video_processing import is_one_face_moving, audio_processing
from ffmpeg import Error as FFmpegError
from vosk import Model, KaldiRecognizer
import wave
import json


def landmarks_for_words(result, landmarks_path, output_path):
    landmarks = np.load(landmarks_path)
    l = 0.04
    if result:
        for i in range(len(result)):
            start_frame = int(np.floor(result[i]['start']/l))
            end_frame = int(np.ceil(result[i]['end']/l))
            output_landmarks_path = os.path.splitext(output_path)[0] + f'_{i}_' + result[i]['word'] + '.npz'
            np.savez(output_landmarks_path, landmarks[start_frame:end_frame+1, 48:60, :])
     
        os.remove(output_path)
    else:
        pass


def create_landmarks_dataset(input_dir, model):
    output_dir = '/content/gdrive/My Drive/landmarks'
    dir_list = os.listdir(input_dir)
    for dir in dir_list:
        subdir = os.path.join(input_dir, dir)
        output_subdir = os.path.join(output_dir, dir)
        if os.path.exists(output_subdir):
            print(dir + ' already checked')
        else:
            os.mkdir(output_subdir)
            files = os.listdir(subdir)
            with tqdm(total=len(files), desc = dir) as pbar:
                for file in files:
                    if os.path.splitext(file)[1] == '.mp4':
                        path_0 = os.path.join(subdir, file)
                        path_1 = os.path.splitext(path_0)[0] + '_2DFull.npy'
                        path_2 = os.path.splitext(path_0)[0] + '_BB.npy'
                        path_3 = os.path.splitext(path_0)[0] + '.txt'
                        output_path = os.path.join(output_subdir, file)
                        if is_one_face_moving(path_2):
                            try:
                                d = audio_processing(path_0, path_3, model)
                                if d:
                                    landmarks_for_words(d, path_0, path_1, output_path)
                                pbar.update(1)
                            except cv2.error as e:
                                continue
                            except AttributeError:
                                continue
                            except KeyError:
                                continue
                            except FFmpegError:
                                continue
                            except FileNotFoundError:
                                continue
                            except EOFError:
                                continue
                        else:
                            continue
