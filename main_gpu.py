import sys
import numpy as np
from scipy.signal import find_peaks, peak_widths
import torch
import torch.nn.functional as F
from pydub import AudioSegment
import plotly.graph_objects as go


def get_args(args):
    file_name = args[0]
    out_file_name = file_name[:-4] + '_gaps_removed'  # f_name + 'gaps_removed'
    length = float(args[1])  # Silence length threshold
    threshold = int(args[2])  # Sound wave power threshold
    try:
        plot = True if int(args[3]) == 1 else False
    except IndexError:
        plot = False
    return file_name, out_file_name, length, threshold, plot


def get_audio_file(file_name):
    sound = AudioSegment.from_mp3(file_name)
    frame_rate = sound.frame_rate
    monos = sound.split_to_mono()
    right = np.array(monos[0].get_array_of_samples())
    left = np.array(monos[1].get_array_of_samples())
    return right, left, frame_rate


def cumulative(right, left):
    '''
    returns the highest amount of audio intensity in each channel
    '''
    return np.where(np.abs(right) > np.abs(left), right, left)


def sil_det(right, left, cumulative, length, frame_rate, threshold):
    frame_len = int(length * frame_rate / 2)
    right, left = torch.tensor(right), torch.tensor(left)
    cumulative = torch.tensor(cumulative)
    thr = torch.where(
        cumulative > threshold, 1, 0).to('cuda', dtype=torch.float32
                                         )
    kernel = torch.ones(1, 1, frame_len).to('cuda')
    thr = F.conv1d(thr.view(1, 1, -1), kernel, padding='same').cpu().view(-1)
    right = (right[thr != 0]).cpu().view(-1, 1)
    left = (left[thr != 0]).cpu().view(-1, 1)
    return np.concatenate((right, left), axis=1), thr


def sqnc_indx(convArr):
    peaks = find_peaks(np.where(convArr > 1, 1, 0))[0]
    width = peak_widths(np.where(convArr > 1, 1, 0), peaks, rel_height=0.5)
    indx = [0]
    for i in range(width[2].shape[0]):
        indx += [int(width[2][i]), int(width[3][i])]
    indx += [convArr.shape[0]-1]
    return indx


def plotter(npArr, threshold, indx, frame_rate):
    x = np.linspace(0, npArr.shape[0], npArr.shape[0])
    fig = go.Figure(data=go.Scatter(x=x, y=npArr))
    fig.add_hline(y=threshold)
    fig.add_hline(y=-threshold)
    for i in range(len(indx)//2):
        fig.add_vrect(
            x0=indx[i*2], x1=indx[i*2+1], fillcolor="red",
            annotation_text=f"{round((indx[i*2+1]-indx[i*2])/frame_rate, 2)}s",
            annotation_position="top left", opacity=0.25, line_width=0)
    fig.show()


def audio_export(output, frame_rate, file_name):
    output = AudioSegment(output.tobytes(),
                          frame_rate=frame_rate,
                          sample_width=2, channels=2)
    output.export(f"{file_name}.mp3", format="mp3")


def main(file_name, out_file_name, length, threshold, plot=False):
    right, left, frame_rate = get_audio_file(file_name)
    cumu = cumulative(right, left)
    out, convArr = sil_det(right, left, cumu, length, frame_rate, threshold)
    indx = sqnc_indx(convArr)
    if plot:
        plotter(cumu, threshold, indx, frame_rate)
        prompter(out, frame_rate, file_name, out_file_name)
    else:
        audio_export(out, frame_rate, out_file_name)


def prompter(out, frame_rate, file_name, out_file_name):
    ans = input('Save a new mp3 file with the detected gaps removed? [y/any] ')
    if ans == 'y':
        audio_export(out, frame_rate, out_file_name)
        return None
    else:
        length = float(input('Enter the new length: '))
        threshold = int(input('Enter the new threshold: '))
        main(file_name, out_file_name, length, threshold, True)


if __name__ == '__main__':
    args = sys.argv[1:]  # "file.mp3" -length -threshold
    file_name, out_file_name, length, threshold, plot = get_args(args)
    main(file_name, out_file_name, length, threshold, plot)
