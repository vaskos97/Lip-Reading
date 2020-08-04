import wave
import json
import ffmpeg
from vosk import Model, KaldiRecognizer
import os
from tqdm import tqdm
import shutil
import glob
from video_preprocessing import *

model =  Model("vosk-model-ru-0.10")
processing('content/gdrive/My Drive/dataset/video/raw', 'processed', model)

create_less_29_word_dir('processed', 'dataset')


