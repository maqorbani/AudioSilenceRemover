import sys
import numpy as np
from scipy.signal import find_peaks, peak_widths
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
    '''
    Silence detection and removal
    length is given in seconds (s)
    Threshold is given in sound wave power not in dB
    '''
    frame_len = int(length * frame_rate / 2)
    conved = np.where(np.abs(cumulative) > threshold, 1, 0)
    conved = np.convolve(np.ones(frame_len), conved, 'same')
    right = right[conved != 0].reshape(-1, 1)
    left = left[conved != 0].reshape(-1, 1)
    return np.concatenate((right, left), axis=1), conved


def sqnc_indx(convArr):
    peaks = find_peaks(np.where(convArr > 1, 1, 0))[0]
    width = peak_widths(np.where(convArr > 1, 1, 0), peaks, rel_height=0.5)
    indx = [0]
    for i in range(width[2].shape[0]):
        indx += [int(width[2][i]), int(width[3][i])]
    indx += [convArr.shape[0]-1]
    return indx


def sampler(npArr, smpl_rate, frame_rate, indx):
    frames = npArr.shape[0]
    dvsn = int(frames // (frame_rate * smpl_rate)) + 1
    rmndr = int((dvsn * frame_rate * smpl_rate) % npArr.shape[0])
    print(frame_rate * smpl_rate)
    indx_ = [i // (frame_rate * smpl_rate) for i in indx]
    smpled = np.pad(npArr, (0, rmndr)).reshape(dvsn, -1)
    smpled = (np.abs(smpled).max(axis=1))
    print(indx_)
    return smpled, indx_


def plotter(npArr, threshold, indx, frame_rate, smpl_rate):
    indx_ = [i * smpl_rate for i in indx]
    x = np.linspace(0, npArr.shape[0]*smpl_rate, npArr.shape[0])
    fig = go.Figure(data=go.Scatter(x=x, y=npArr))
    fig.add_hline(y=threshold)
    fig.add_hline(y=-threshold)
    for i in range(len(indx_)//2):
        fig.add_vrect(
            x0=indx_[i*2], x1=indx_[i*2+1], fillcolor="red",
            annotation_text=f"{round((indx_[i*2+1]-indx_[i*2]), 2)}s",
            annotation_position="top left", opacity=0.25, line_width=0)
    fig.show()


def audio_export(output, frame_rate, file_name):
    output = AudioSegment(output.tobytes(),
                          frame_rate=frame_rate,
                          sample_width=2, channels=2)
    output.export(f"{file_name}.mp3", format="mp3")


def main(file_name, out_file_name, length, threshold, plot=False):
    right, left, frame_rate = get_audio_file(file_name)
    smpl_rate = 0.01  # ->user input
    cumu = cumulative(right, left)
    out, conved = sil_det(right, left, cumu, length, frame_rate, threshold)
    indx = sqnc_indx(conved)
    if plot:
        sampled, indx_ = sampler(cumu, smpl_rate, frame_rate, indx)
        plotter(sampled, threshold, indx_, frame_rate, smpl_rate)
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
    args = sys.argv[1:]  # "file.mp3" length threshold plot_toggle
    file_name, out_file_name, length, threshold, plot = get_args(args)
    main(file_name, out_file_name, length, threshold, plot)
