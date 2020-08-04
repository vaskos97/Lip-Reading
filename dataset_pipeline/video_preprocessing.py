import os
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import nltk
import string
from vosk import Model, KaldiRecognizer
import os
import wave
import json
import ffmpeg
from ffmpeg import Error as FFmpegError


def distance(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5



def is_one_face_moving(path):
    bb = np.load(path)
    FLAG = True
    if ((bb[0] == bb[len(bb)-1]).all()):
        FLAG = False
        return FLAG
    for i in range(len(bb)-1):
        current = bb_intersection_over_union(bb[i], bb[i+1])
        if current < 0.8:
            FLAG = False
            break
    return FLAG



def align(image, leftMouthEdge, rightMouthEdge):
    # compute the angle between the mouth edges
    dY = rightMouthEdge[1] - leftMouthEdge[1]
    dX = rightMouthEdge[0] - leftMouthEdge[0]
    angle = np.degrees(np.arctan2(dY, dX))


    # compute center (x, y)-coordinates (i.e., the median point)
    mouthCenter = ((leftMouthEdge[0] + rightMouthEdge[0]) // 2,
          (leftMouthEdge[1] + rightMouthEdge[1]) // 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(mouthCenter, angle, 1)

    # apply the affine transformation
    output = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # return the aligned face
    return output


def cropping(input_video_path, landmarks_path, bb_path, output_path, desired_size):
    frames = []
    frames_tracked=[]
  
    landmarks = np.load(landmarks_path)
    bb = np.load(bb_path)
    count = 0
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = landmarks.shape[0]
    while cap.isOpened():
        ret, frame = cap.read()
        frames.append(frame)
        count = count + 1
        if (count > (video_length-1)):
            cap.release()

    for j, frame in enumerate(frames):

        left_mouth_edge, right_mouth_edge = bb[j,:2] + landmarks[j, 48, :], bb[j,:2] + landmarks[j, 54, :]
        nose_tip = bb[j,:2] + landmarks[j, 30, :]
    
        new_frame = align(frame, left_mouth_edge, right_mouth_edge)

        horizontal_dist = 1.05*right_mouth_edge[0] - 0.95*left_mouth_edge[0]
        center = [(left_mouth_edge[0] + right_mouth_edge[0]) // 2,
        (left_mouth_edge[1] + right_mouth_edge[1]) // 2]

        d_mn = distance(center, nose_tip)

        w = min(3*d_mn, max(1.5*d_mn, horizontal_dist))

        x1 = int(center[0] - w/2)
        x2 = int(center[0] + w/2)
        y1 = int(center[1] - w/2)
        y2 = int(center[1] + w/2)
        roi = new_frame[y1:y2, x1:x2]
        resized_frame = cv2.resize(roi, (desired_size, desired_size)) 
    
        frames_tracked.append(resized_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
    video_tracked = cv2.VideoWriter(output_path, fourcc, fps, (desired_size, desired_size))
    for frame in frames_tracked:
        video_tracked.write(frame)
    video_tracked.release()


def save(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0 
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        frames.append(frame)
        count += 1
        if (count > (video_length-1)):
            cap.release()

    np.savez(os.path.splitext(video_path)[0] + '.npz', frames)


def audio_processing(path_0, model):
    in_stream = ffmpeg.input(path_0)
    audio_stream = in_stream.audio
    audio_name = os.path.splitext(path_0)[0] + '.wav'
    out_stream = ffmpeg.output(audio_stream, audio_name, ab=160, ac=1, ar=44100)
    ffmpeg.run(out_stream)
    wf = wave.open(audio_name, "rb") 
    rec = KaldiRecognizer(model, wf.getframerate())

    while True:
        data = wf.readframes(10000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    res = json.loads(rec.FinalResult())

    if res['text'] != '':
        return res['result']
    else:
        return False


def video_processing(path_0, path_1, path_2, output_path, result, size):
    if result:
        cropping(path_0, path_1, path_2, output_path, size)
        in_file = ffmpeg.input(output_path)
        for i in range(len(result)):
            output_video_path = os.path.splitext(output_path)[0] + f'_{i}_' + result[i]['word'] + '.mp4'
            in_file.trim(start=result[i]['start'], end=result[i]['end']).output(output_video_path).run()
    else:
        pass


def processing(input_dir, output_dir, model):
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
                        output_path = os.path.join(output_subdir, file)
                        if is_one_face_moving(path_2):
                            try:
                                d =audio_processing(path_0, model)
                                if d:
                                    video_processing(path_0, path_1, path_2, output_path, d, 96)
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
                        else:
                            continue
            

def width_of_lip_region(landmarks_path, bb_path):
    landmarks = np.load(landmarks_path)
    bb = np.load(bb_path)
    left_mouth_edge, right_mouth_edge = bb[0,:2] + landmarks[0, 48, :], bb[0,:2] + landmarks[0, 54, :]
    w = right_mouth_edge[0] - left_mouth_edge[0]
    return w


def stat_counting(input_dir, current_list):
    dir_list = os.listdir(input_dir)
    with tqdm(total=len(dir_list)) as pbar:
        for dir in dir_list:
            subdir = os.path.join(input_dir, dir)
            files = os.listdir(subdir)
            for file in files:
                if os.path.splitext(file)[1] == '.mp4':
                    path_0 = os.path.join(subdir, file)
                    path_1 = os.path.splitext(path_0)[0] + '_2DFull.npy'
                    path_2 = os.path.splitext(path_0)[0] + '_BB.npy'
                    current_list.append(width_of_lip_region(path_1, path_2))
            pbar.update(1)
