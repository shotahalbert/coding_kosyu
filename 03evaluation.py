
# coding: utf-8

import re
import os
import gc
from datetime import datetime
from pipetools import pipe, X
import numpy as np
from sklearn.externals import joblib
import json
import matplotlib.cm as cm
import warnings
from matplotlib import pylab as plt
plt.style.use("ggplot")
warnings.filterwarnings("ignore")


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def make_dirs():
    make_dir("../data/test_data/processed/")
    make_dir("../result/model/")
    make_dir("../result/summary/")
    make_dir("../result/anomaly_level/")
    make_dir("../result/anomaly_border/")


def gather_csv(csv_dir, dict_IP):
    import_csv_list = os.listdir(csv_dir)

    for csv_name in import_csv_list:
        root, ext = os.path.splitext(csv_name)
        if os.path.isdir(csv_name) or ext != ".csv":
            continue

        sensorIP = csv_name > pipe | (re.split, r"\_") | X[0]

        if sensorIP in dict_IP.keys():
            file_names = dict_IP[sensorIP]
            if csv_name not in file_names:
                file_names.append(csv_name)
                dict_IP[sensorIP] = file_names
        else:
            file_names = []
            file_names.append(csv_name)
            dict_IP[sensorIP] = file_names

    return dict_IP


def read_data(file_name):
    data = np.genfromtxt(file_name, delimiter=",", usecols=range(0, 3))
    data = data[~np.isnan(data).any(axis=1)]
    return data


def check_datas(key, dict_test_IP):
    b_ret = True

    if len(dict_test_IP[key]) == 0 \
       or not os.path.exists("../result/model/" + key + ".json") \
       or not os.path.exists("../result/model/" + key + ".pkl") \
       or not os.path.exists("../result/model/" + key + "_anomaly.csv"):
        b_ret = False

    return b_ret


def draw_anomaly_level(data, dict_threshold, file_name, n_state):
    key_threshold = list(dict_threshold.keys())
    key_threshold.sort()
    val_threshold = list(dict_threshold.values())
    val_threshold.sort()
    len_val = len(val_threshold)
    colors = np.linspace(0, 0.9, len_val+1)
    x = range(len(data))
    y = data

    fig = plt.figure(figsize=(20, 7))
    fig.suptitle("time series anomaly level\n \
                 Original Evaluate File: %s\nstates: %d"
                 % (file_name, n_state))
    ax = fig.add_subplot(111)
    for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]):
        drawed = False
        for thld_idx in range(0, len_val+1):
            idx = len_val - thld_idx - 1
            if max(y1, y2) > val_threshold[idx]:
                ax.plot([x1, x2],
                        [y1, y2],
                        linestyle="solid",
                        lw=1.5,
                        alpha=0.7,
                        color=cm.nipy_spectral(colors[idx+1]))
                drawed = True
                break
        if not drawed:
            ax.plot([x1, x2],
                    [y1, y2],
                    linestyle="solid",
                    lw=1.5,
                    alpha=0.7,
                    color=cm.nipy_spectral(colors[0]))

    xmin, xmax = ax.get_xlim()
    for key in key_threshold:
        ratio_over_thld = \
            float(len(np.where(data >= dict_threshold[key])[0])) \
            / float(len(data))
        label = "thld(%.3f) = %.3f\nPr(>=%.3f) = %.4f" \
            % (float(key),
               dict_threshold[key],
               dict_threshold[key],
               ratio_over_thld)
        ax.hlines(dict_threshold[key],
                  xmin=xmin,
                  xmax=xmax,
                  linestyle="dashed",
                  lw=1,
                  label=label)
    ax.legend()
    ax.set_ylabel("anomaly")
    root, ext = os.path.splitext(file_name)
    output_filename = os.path.join("../result/anomaly_level/",
                                   root
                                   + ".png")
    fig.savefig(output_filename)
    plt.close(fig)

    plt.cla()
    plt.clf()
    del ax
    del fig


def draw_anomaly_border(data_anomaly_learn,
                        data_anomaly_eval,
                        file_name, n_state):
    detection_border = np.array([[0., 1.]])
    for idx in range(1, 101):
        thld = np.percentile(data_anomaly_learn, idx)
        ratio_eval_over_thld = \
            float(len(np.where(data_anomaly_eval >= thld)[0])) \
            / float(len(data_anomaly_eval))
        detection_border = np.append(detection_border,
                                     [[float(idx)/100.,
                                      ratio_eval_over_thld]],
                                     axis=0)

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle("percent for threshold percentile - "
                 + "percent of anomaly over than threshold\n"
                 + "Original Evaluate File: %s\nstate: %d"
                 % (file_name, n_state))
    ax = fig.add_subplot(111)
    ax.plot(detection_border[:, 0], detection_border[:, 1], lw=1)
    major_ticks = np.arange(0, 1.01, 0.1)
    minor_ticks = np.arange(0, 1.01, 0.01)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which="both")
    ax.grid(which="minor", alpha=0.5, linestyle="dashed")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("percent for threshold percentile")
    ax.set_ylabel("percent of anomaly over than threshold")

    root, ext = os.path.splitext(file_name)
    output_filename = os.path.join("../result/anomaly_border/",
                                   root
                                   + ".png")
    fig.savefig(output_filename)
    plt.close(fig)

    plt.cla()
    plt.clf()
    del ax
    del fig

    output_filename = os.path.join("../result/anomaly_border/",
                                   root
                                   + ".csv")
    np.savetxt(output_filename, detection_border, delimiter=",", fmt="%.8f")


# @profile
def do_detect(dict_test_IP):

    for key in dict_test_IP.keys():
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print("IP:", key)

        if not check_datas(key, dict_test_IP):
            print("lack of some files : ", key)
            continue

        try:
            model = joblib.load("../result/model/" + key + ".pkl")
            f_threshold = open("../result/model/" + key + ".json", "r")
            dict_threshold = json.load(f_threshold)
            data_anomaly_learn = np.genfromtxt("../result/model/"
                                               + key
                                               + "_anomaly.csv",
                                               delimiter=",",
                                               usecols=range(0, 1))
            data_anomaly_learn = \
                data_anomaly_learn[~np.isnan(data_anomaly_learn)]

        except:
            print("failed to load model data : ", key)
            continue

        model_state = np.shape(model.means_)[0]
        print("State:", model_state)

        idx = 1
        for file_eval in dict_test_IP[key]:
            print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "eval [%d/%d]"
                  % (idx, len(dict_test_IP[key])))
            data_eval = read_data(os.path.join("../data/test_data/processed/",
                                               file_eval))

            try:
                _, state_sequence = model.decode(data_eval, algorithm="map")
                logprob = model._compute_log_likelihood(data_eval)

            except:
                print("except occured with model")
                continue

            logprob_sequence = list((logprob[x, state_sequence[x]])
                                    for x in range(len(state_sequence)))
            data_anomaly_eval = -1 * (np.array(logprob_sequence))

            draw_anomaly_level(data_anomaly_eval,
                               dict_threshold,
                               file_eval,
                               model_state)
            draw_anomaly_border(data_anomaly_learn,
                                data_anomaly_eval,
                                file_eval, model_state)

            idx += 1
            gc.collect()
            gc.collect()

        gc.collect()
        gc.collect()

    print("---- detect DONE ----")


# @profile
def main():
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "main() start")
    make_dirs()
    dict_test_IP = {}
    dict_test_IP = gather_csv("../data/test_data/processed/", dict_test_IP)
    do_detect(dict_test_IP)

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "main() end")


if __name__ == "__main__":
    main()
