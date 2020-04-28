
# coding: utf-8

import numpy as np
import re
import os
from datetime import datetime
from pipetools import pipe, X
from scipy import fftpack, hamming
from matplotlib import pylab as plt
import warnings
warnings.filterwarnings("ignore")


def get_threshold(data, percentile, weight):
    threshold_upper = np.percentile(data, 50) \
        + abs(np.percentile(data, percentile)
              - np.percentile(data, 50)) * weight
    threshold_lower = np.percentile(data, 50) \
        - abs(np.percentile(data, (100-percentile))
              - np.percentile(data, 50)) * weight
    return threshold_upper, threshold_lower


def filter_threshold(data, upper, lower, data_idx, threshold_steps):
    idx = threshold_steps
    del_samples = 0
    while idx < len(data) - threshold_steps - 1:
        do_del = False
        if(upper < data[idx, data_idx]):
            if(upper > max(data[(idx-threshold_steps):idx, data_idx]) and
               upper > max(data[(idx+1):(idx+threshold_steps), data_idx])): #一時的  このdataを変数にする?
                do_del = True
        if(lower > data[idx, data_idx]):
            if(lower < min(data[(idx-threshold_steps):idx, data_idx]) and
               lower < min(data[(idx+1):(idx+threshold_steps), data_idx])):
                do_del = True
        if(do_del):
                data = np.delete(data, idx, axis=0)
                idx += -1
                del_samples += 1
        idx += 1
    return data, del_samples


def save_image(dataRaw, dataFilter,
               upper_x, lower_x,
               upper_y, lower_y,
               upper_z, lower_z,
               dir_data, csv_name,
               del_samples_x,
               del_samples_y,
               del_samples_z): #引数が大きすぎる気がする。del_samplesはセットで考えた方が綺麗に見えないかな?

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Diff Filtered file: %s \n Delete times X:%d Y:%d Z:%d" %
                 (csv_name, del_samples_x, del_samples_y, del_samples_z)) #ここから87ページまでは辞書とfor文を使って解決できないか?
    ax_raw_x = fig.add_subplot(231)
    ax_raw_y = fig.add_subplot(232)
    ax_raw_z = fig.add_subplot(233)
    ax_fil_x = fig.add_subplot(234, sharey=ax_raw_x)
    ax_fil_y = fig.add_subplot(235, sharey=ax_raw_y)
    ax_fil_z = fig.add_subplot(236, sharey=ax_raw_z)
    ax_raw_x.plot(range(len(dataRaw)), dataRaw[:, 0], "b-", lw=1)
    ax_raw_y.plot(range(len(dataRaw)), dataRaw[:, 1], "r-", lw=1)
    ax_raw_z.plot(range(len(dataRaw)), dataRaw[:, 2], "g-", lw=1)
    ax_raw_x.hlines([upper_x, lower_x],
                    xmin=0, xmax=len(dataRaw),
                    linestyle="dashed", lw=1)
    ax_raw_y.hlines([upper_y, lower_y],
                    xmin=0, xmax=len(dataRaw),
                    linestyle="dashed", lw=1)
    ax_raw_z.hlines([upper_z, lower_z],
                    xmin=0, xmax=len(dataRaw),
                    linestyle="dashed", lw=1)
    ax_fil_x.plot(range(len(dataFilter)), dataFilter[:, 0], "b-", lw=1)
    ax_fil_y.plot(range(len(dataFilter)), dataFilter[:, 1], "r-", lw=1)
    ax_fil_z.plot(range(len(dataFilter)), dataFilter[:, 2], "g-", lw=1)
    ax_fil_x.hlines([upper_x, lower_x],
                    xmin=0, xmax=len(dataFilter),
                    linestyle="dashed", lw=1)
    ax_fil_y.hlines([upper_y, lower_y],
                    xmin=0, xmax=len(dataFilter),
                    linestyle="dashed", lw=1)
    ax_fil_z.hlines([upper_z, lower_z],
                    xmin=0, xmax=len(dataFilter),
                    linestyle="dashed", lw=1)

    root, ext = os.path.splitext(csv_name)
    output_filename = os.path.join(dir_data +
                                   "filter/",
                                   root + ".png")
    fig.savefig(output_filename)
    plt.close(fig)

    plt.cla()
    plt.clf()
    del fig


def calcfft(data, sampling_rate):
    # DC component remove
    data = data[:] - np.mean(data)
    data = data * hamming(len(data))  #済 一時変数hamm_windowの削除
    sample_freq = fftpack.fftfreq(data[:].size, 1./float(sampling_rate))
    y_fft = fftpack.fft(data[:])
    pidxs = np.where(sample_freq > 0)

    _freqs, powers = sample_freq[pidxs], 10. * \
        np.log10(np.abs(y_fft)[pidxs] ** 2)
    del _freqs
    return powers


def fft_time_series(dataFilter,
                    samplinglen,
                    interval,
                    sampling_rate,
                    dir_data,
                    csv_name):

    maxpowers = []
    for i in range(0, ((len(dataFilter)-samplinglen)//interval+1)):
        tmppowerX = calcfft(
            dataFilter[(0+i*interval):(samplinglen+i*interval), 0],
            sampling_rate)
        tmppowerY = calcfft(
            dataFilter[(0+i*interval):(samplinglen+i*interval), 1],
            sampling_rate)
        tmppowerZ = calcfft(
            dataFilter[(0+i*interval):(samplinglen+i*interval), 2],
            sampling_rate)
        maxpowers.append([max(tmppowerX), max(tmppowerY), max(tmppowerZ)]) # 上と同様for文箇所

    maxpowers = np.array(maxpowers)

    # return maxpowers

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("FFT max power time series\n file :%s" % csv_name)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.plot(range(len(maxpowers[:, 0])), maxpowers[:, 0], "b-", lw=1)
    ax2.plot(range(len(maxpowers[:, 0])), maxpowers[:, 1], "b-", lw=1)
    ax3.plot(range(len(maxpowers[:, 0])), maxpowers[:, 2], "b-", lw=1)
    ax1.set_ylabel("PowerX")
    ax2.set_ylabel("PowerY")
    ax3.set_ylabel("PowerZ")
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    root, ext = os.path.splitext(csv_name)
    output_filename = os.path.join(dir_data, root + ".png")
    fig.savefig(output_filename)
    plt.close(fig)

    plt.cla()
    plt.clf()
    del fig

    return maxpowers


def gather_csv(csv_dir, dict_IP):
    import_csv_list = os.listdir(csv_dir)
    for csv_name in import_csv_list:
        root, ext = os.path.splitext(csv_name)
        if os.path.isdir(csv_name) or ext != ".csv":
            continue

        sensorIP = csv_name > pipe | (re.split, r"\_") | X[0]

        if(sensorIP in dict_IP.keys()):
            file_names = dict_IP[sensorIP]
            if csv_name not in file_names:
                file_names.append(csv_name)
                dict_IP[sensorIP] = file_names
        else:
            file_names = []
            file_names.append(csv_name)
            dict_IP[sensorIP] = file_names

    return dict_IP


def write_filter_data(dir_in,
                      dir_out,
                      import_csv_list,
                      threshold_steps,
                      percentile,
                      weight):

    for csv_name in import_csv_list:
        dataRaw = np.genfromtxt(
            os.path.join(dir_in, csv_name),
            delimiter=",", usecols=range(0, 3))

        # filter nan
        dataRaw = dataRaw[~np.isnan(dataRaw).any(axis=1)]

        if len(dataRaw) < 256:
            print("samples too small : ", csv_name)
            continue

        dataFilter = dataRaw

        # distance from median
        threshold_upper_x, threshold_lower_x = \
            get_threshold(dataFilter[:, 0], percentile, weight)
        threshold_upper_y, threshold_lower_y = \
            get_threshold(dataFilter[:, 1], percentile, weight)
        threshold_upper_z, threshold_lower_z = \
            get_threshold(dataFilter[:, 2], percentile, weight) #辞書for dataFilterはいちいち中身が変わっている
        
        dataFilter, del_samples_x = \
            filter_threshold(dataFilter,
                             threshold_upper_x,
                             threshold_lower_x,
                             0,
                             threshold_steps)
        dataFilter, del_samples_y = \
            filter_threshold(dataFilter,
                             threshold_upper_y,
                             threshold_lower_y,
                             1,
                             threshold_steps)
        dataFilter, del_samples_z = \
            filter_threshold(dataFilter,
                             threshold_upper_z,
                             threshold_lower_z,
                             2,
                             threshold_steps)

        root, ext = os.path.splitext(csv_name)
        output_filename = os.path.join(dir_out + "filter/", root + ".csv")
        np.savetxt(output_filename, dataFilter, delimiter=",")

        samplinglen = 256
        interval = 32
        sampling_rate = 200
        data_Powers = fft_time_series(dataFilter,
                                      samplinglen,
                                      interval,
                                      sampling_rate,
                                      dir_out,
                                      csv_name)

        output_filename = os.path.join(dir_out, root + ".csv")
        np.savetxt(output_filename, data_Powers, delimiter=",")

        save_image(dataRaw, dataFilter,
                   threshold_upper_x, threshold_lower_x,
                   threshold_upper_y, threshold_lower_y,
                   threshold_upper_z, threshold_lower_z,
                   dir_out, csv_name,
                   del_samples_x,
                   del_samples_y,
                   del_samples_z)


def main():
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "main() start")
    threshold_steps = 50
    percentile = 97
    weight = 3.0

    list_data = ["learn_data","evaluate_data","test_data"] #3つの処理をlistforで回した
    for datatype in list_data: 
        print(datatype)
        dir_in = "../data/" + datatype + "/raw_data/"
        dir_out= "../data/" + datatype + "/processed/"
        if not os.path.isdir(dir_in):
            os.mkdir(dir_in)
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        dict_IP = {}
        dict_IP = gather_csv(dir_in, dict_IP)
        for key in dict_IP.keys():
            print(key)
            write_filter_data(dir_in,
                            dir_out,
                            dict_IP[key],
                            threshold_steps,
                            percentile,
                            weight)
            
    
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "main() end")


if __name__ == '__main__':
    main()
