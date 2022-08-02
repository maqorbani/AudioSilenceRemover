import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play


def sil_det(npArr, npArr2, length, frame_rate, threshold):
    '''
    Silence detection and removal
    length is given in seconds (s)
    Threshold is given in sound wave power not in dB
    '''
    frame_len = int(length * frame_rate)
    a = np.where(np.abs(npArr) > threshold, 1, 0)
    a = np.convolve(np.ones(frame_len), a, 'same')
    npArr = npArr[a != 0].reshape(-1, 1)
    npArr2 = npArr2[a != 0].reshape(-1, 1)
    return np.concatenate((npArr, npArr2), axis=1)


def sil_det_gpu(npArr, npArr2, length, frame_rate, threshold):
    frame_len = int(length * frame_rate)
    npArr = torch.tensor(npArr)
    npArr2 = torch.tensor(npArr2)
    thr = torch.where(npArr > threshold, 1, 0).to('cuda', type=torch.float32)
    kernel = torch.ones(1, 1, frame_len).to('cuda')
    thr = F.conv1d(thr.view(1, 1, -1), kernel, padding='same').cpu().view(-1)
    npArr = (npArr[thr != 0]).cpu().view(-1, 1)
    npArr2 = (npArr2[thr != 0]).cpu().view(-1, 1)
    return (npArr[thr != 0]).cpu()


if __name__ == '__main__':
    pass
