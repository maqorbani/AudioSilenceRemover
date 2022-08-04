import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pydub import AudioSegment
# from pydub.playback import play


def sil_det(npArr, npArr2, length, frame_rate, threshold):
    '''
    Silence detection and removal
    length is given in seconds (s)
    Threshold is given in sound wave power not in dB
    '''
    frame_len = int(length * frame_rate / 2)
    a = np.where(np.abs(npArr) > threshold, 1, 0)
    a = np.convolve(np.ones(frame_len), a, 'same')
    npArr = npArr[a != 0].reshape(-1, 1)
    npArr2 = npArr2[a != 0].reshape(-1, 1)
    return np.concatenate((npArr, npArr2), axis=1)


def sil_det_gpu(npArr, npArr2, length, frame_rate, threshold):
    frame_len = int(length * frame_rate / 2)
    npArr = torch.tensor(npArr)
    npArr2 = torch.tensor(npArr2)
    thr = torch.where(npArr > threshold, 1, 0).to('cuda', type=torch.float32)
    kernel = torch.ones(1, 1, frame_len).to('cuda')
    thr = F.conv1d(thr.view(1, 1, -1), kernel, padding='same').cpu().view(-1)
    npArr = (npArr[thr != 0]).cpu().view(-1, 1)
    npArr2 = (npArr2[thr != 0]).cpu().view(-1, 1)
    return (npArr[thr != 0]).cpu()


def get_audio_file(file_name):
    sound = AudioSegment.from_mp3(file_name)
    frame_rate = sound.frame_rate
    monos = sound.split_to_mono()
    right = np.array(monos[0].get_array_of_samples())
    left = np.array(monos[1].get_array_of_samples())
    return right, left, frame_rate


def audio_export(output, frame_rate, file_name):
    output = AudioSegment(out.tobytes(),
                          frame_rate=frame_rate,
                          sample_width=2, channels=2)
    output.export(f"{file_name}.mp3", format="mp3")


if __name__ == '__main__':
    args = sys.argv[1:]  # "file.mp3" -length -threshold
    try:
        length = args[1]  # Silence length threshold
    except ValueError:
        length = 0.5  # Default length
    try:
        threshold = args[2]  # Sound wave power threshold
    except ValueError:
        threshold = 3000  # Default threshold
    file_name = args[0]
    out_file_name = file_name[:-4] + '_gaps_removed'
    right, left, frame_rate = get_audio_file(file_name)
    out = sil_det(right, left, length, frame_rate, threshold)
    audio_export(out, frame_rate, out_file_name)
