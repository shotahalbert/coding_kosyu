
# coding: utf-8

import re
import os
import gc
from datetime import datetime
from pipetools import pipe, X
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from hmmlearn import hmm
import json
from matplotlib import pylab as plt
import matplotlib.cm as cm
import warnings
plt.style.use("ggplot")
warnings.filterwarnings("ignore")


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def make_dirs():
    make_dir("../data/learn_data/processed/")
    make_dir("../data/evaluate_data/processed/")
    make_dir("../result/loglikelihood/")
    make_dir("../result/model/")
    make_dir("../result/summary/")


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


def combine_learn_data(files):
    for idx in range(0, len(files)):
        data = read_data(os.path.join("../data/learn_data/processed",
                                      files[idx]))

        if idx == 0:
            ret_data = data
            ret_len = [len(data)]
        else:
            ret_data = np.append(ret_data, data, axis=0)
            ret_len.extend([len(data)])

    return ret_data, ret_len


def gen_columns(model_states):
    lab_states = [("nstates_" + str(x)) + "_per_seq" for x in model_states]
    columns = ["IP", "seq_length_learn", "seq_length_eval", "file_eval"]
    columns.extend(lab_states)

    lab_start = []
    for states in model_states:
        for i in range(1, states+1):
            lab_start.append("start_" + str(i) + "_of_" + str(states))

    lab_trans = []
    for states in model_states:
        for i in range(1, states+1):
            for j in range(1, states+1):
                lab_trans.append("trans_"
                                 + str(i)
                                 + "_to_"
                                 + str(j)
                                 + "_of_"
                                 + str(states))

    columns_model = ["IP"]
    columns_model.extend(lab_start)
    columns_model.extend(lab_trans)

    return columns, columns_model


def draw_likelihood(key, model_states, ary_likelihood):
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Log-Likelihood per sample\nIP: %s" % (key))
    ax = fig.add_subplot(111)
    colors = np.linspace(0, 1, len(ary_likelihood))
    for idx in range(0, len(ary_likelihood)):
        ax.plot(model_states,
                ary_likelihood[idx, :],
                color=cm.brg(colors[idx]),
                lw=3)
    ax.set_xticks(model_states)
    ax.set_xlabel("States")
    ax.set_ylabel("Log-Likelihood")

    output_filename = os.path.join("../result/loglikelihood/",
                                   key + "_log_likelihood.png")
    fig.savefig(output_filename)
    plt.close(fig)


def check_datas(key, dict_learn_IP, dict_eval_IP):
    b_ret = True

    if len(dict_learn_IP[key]) == 0 \
       or key not in dict_eval_IP \
       or len(dict_eval_IP[key]) == 0:
        b_ret = False

    return b_ret


def do_learn(model_states, idx_state, key, dict_learn_IP, list_model):
    learns = dict_learn_IP[key]
    data_learn, data_len = combine_learn_data(learns)
    n_state = model_states[idx_state]
    n_iter = 100
    model = hmm.GaussianHMM(n_components=n_state,
                            covariance_type="full",
                            n_iter=n_iter)
    np.random.seed(123)

    try:
        model.fit(data_learn, data_len)
    except:
        print("except occured with fit()")
        return None, list_model, data_len, False

    start_idx = 1 + sum(model_states[:idx_state])
    trans_idx = 1 + sum(model_states) \
                  + sum(np.array(model_states[: idx_state]) ** 2)
    list_model[start_idx:(start_idx + n_state)] = model.startprob_.tolist()
    list_model[trans_idx:(trans_idx + n_state * n_state)] = \
        np.reshape(model.transmat_, (1, -1)).tolist()[0]
    return model, list_model, data_len, True


def draw_anomaly_histgram(anomaly_level, key, n_state, dict_threshold):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("histgram of anomaly level\nIP: %s states: %d"
                 % (key, n_state))
    ax = fig.add_subplot(111)
    df = pd.DataFrame(anomaly_level)
    df.columns = ["data"]
    df.plot(bins=50, alpha=0.5, figsize=(15, 10), kind="hist", ax=ax)
    ymin, ymax = ax.get_ylim()
    colors = np.linspace(0, 1, len(dict_threshold.keys()))
    color_idx = 0
    for threshold in dict_threshold:
        label = "thld(%.3f) = %.3f" % (threshold, dict_threshold[threshold])
        color = cm.winter(colors[color_idx])
        ax.vlines(dict_threshold[threshold],
                  ymin=ymin,
                  ymax=ymax,
                  linestyle="solid",
                  lw=2,
                  color=color,
                  label=label)
        color_idx += 1
    ax.set_xlabel("anomaly")
    ax.legend()
    output_filename = os.path.join("../result/model/", key + "_anomaly.png")
    fig.savefig(output_filename)
    plt.close(fig)


def get_threshold(model, key, dict_learn_IP, ratio_fault_threshold, n_state):
    anomaly_level = []
    for file in dict_learn_IP[key]:
        data = read_data(os.path.join("../data/learn_data/processed/", file))

        try:
            _, state_sequence = model.decode(data, algorithm="map")
            logprob = model._compute_log_likelihood(data)

        except:
            print("except occured with model")
            continue

        logprob_sequence = list((logprob[x, state_sequence[x]])
                                for x in range(len(state_sequence)))
        anomaly_level.extend(-1 * (np.array(logprob_sequence)))

    if not anomaly_level:
        return {}, False

    dict_threshold = {}
    for ratio in ratio_fault_threshold:
        dict_threshold[ratio] = np.percentile(np.array(anomaly_level),
                                              float(100*(1-ratio)))

    anomaly_level = np.sort(anomaly_level)

    return dict_threshold, anomaly_level, True


def do_evaluate(model,
                model_states,
                idx_state,
                key,
                dict_eval_IP,
                len_data_learn,
                columns_eval,
                list_nan_eval,
                summary_eval,
                likelihood_per_sample):
    evals = dict_eval_IP[key]

    idx = 1
    likelihoods = []
    for file_eval in evals:
        data_eval = read_data(os.path.join("../data/evaluate_data/processed/",
                              file_eval))

        try:
            likelihood = model.score(data_eval)

        except:
            print("except occured with model")
            continue

        likelihood = likelihood / float(len(data_eval))
        likelihoods.append(likelihood)

        df = summary_eval[summary_eval[columns_eval[3]].isin([file_eval])]

        if len(df) == 0:
            list_new = [key, len_data_learn, len(data_eval), file_eval]
            list_new.extend(list_nan_eval)
            summary_eval = summary_eval.append(
                pd.Series(list_new, index=summary_eval.columns),
                ignore_index=True)

        summary_eval.loc[summary_eval.file_eval == file_eval,
                         columns_eval[4+idx_state]
                         ] = likelihood

        idx += 1

    likelihood_per_sample = np.append(likelihood_per_sample,
                                      np.array([likelihoods]),
                                      axis=0)
    return summary_eval, likelihood_per_sample


# @profile
def do_learn_and_evaluate(model_states,
                          dict_learn_IP,
                          dict_eval_IP,
                          ratio_fault_threshold):
    columns_eval, columns_model = gen_columns(model_states)

    list_nan_eval = [0] * len(model_states)
    list_nan_model = [0] * (len(columns_model) - 1)

    for key in dict_learn_IP.keys():
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print("IP:", key)

        summary_eval = pd.DataFrame(columns=columns_eval)
        summary_model = pd.DataFrame(columns=columns_model)

        if not check_datas(key, dict_learn_IP, dict_eval_IP):
            print("lack of some files : ", key)
            continue

        likelihood_per_sample = np.empty((0, len(dict_eval_IP[key])), float)

        list_model = [key]
        list_model.extend(list_nan_model)
        failed = False
        first_model = True
        for idx_state in range(0, len(model_states)):
            print("State:", model_states[idx_state])

            model, list_model, data_len, successed = do_learn(model_states,
                                                              idx_state,
                                                              key,
                                                              dict_learn_IP,
                                                              list_model)

            if not successed:
                failed = True
                continue

            dict_anomaly_thresholds, anomaly_data, successed = \
                get_threshold(model,
                              key,
                              dict_learn_IP,
                              ratio_fault_threshold,
                              model_states[idx_state])

            if not successed:
                failed = True
                continue

            summary_eval, likelihood_per_sample = \
                do_evaluate(model,
                            model_states,
                            idx_state,
                            key,
                            dict_eval_IP,
                            sum(data_len),
                            columns_eval,
                            list_nan_eval,
                            summary_eval,
                            likelihood_per_sample)
            mean_likelihood = np.mean(likelihood_per_sample[idx_state, :])

            if first_model:
                best_model = model
                best_thld = dict_anomaly_thresholds
                best_anomaly = anomaly_data
                best_likelihood = mean_likelihood
                first_model = False
            else:
                if best_likelihood < mean_likelihood:
                    del best_model
                    best_model = model
                    best_thld = dict_anomaly_thresholds
                    best_anomaly = anomaly_data
                    best_likelihood = mean_likelihood

            gc.collect()
            gc.collect()

        if not failed:
            draw_likelihood(key, model_states, likelihood_per_sample.T)
        else:
            file_path = os.path.join("../result/loglikelihood/",
                                     key + "_log_likelihood.png")
            print("cannot draw graph cause of some error : ", file_path)

        if first_model:
            print("Failed to make any model : ", key)
            gc.collect()
            gc.collect()
            continue

        print(key, "best state :", np.shape(best_model.means_)[0])
        joblib.dump(best_model, "../result/model/" + key + ".pkl")
        f_json = open("../result/model/" + key + ".json", "w")
        json.dump(best_thld, f_json)
        draw_anomaly_histgram(best_anomaly,
                              key,
                              np.shape(best_model.means_)[0],
                              best_thld)
        np.savetxt(os.path.join("../result/model/" + key + "_anomaly.csv"),
                   best_anomaly, delimiter=",")

        summary_model = summary_model.append(
            pd.Series(list_model, index=summary_model.columns),
            ignore_index=True)
        summary_eval.to_csv("../result/summary/summary_eval_" + key + ".csv")
        summary_model.to_csv("../result/summary/summary_model_" + key + ".csv")

        gc.collect()
        gc.collect()


# @profile
def main():
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "main() start")
    model_states = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    ratio_anomaly_threshold = [0.05, 0.01, 0.001]
    make_dirs()
    dict_learn_IP = {}
    dict_learn_IP = gather_csv("../data/learn_data/processed/", dict_learn_IP)
    dict_eval_IP = {}
    dict_eval_IP = gather_csv("../data/evaluate_data/processed/", dict_eval_IP)

    do_learn_and_evaluate(model_states,
                          dict_learn_IP,
                          dict_eval_IP,
                          ratio_anomaly_threshold)

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "main() end")


if __name__ == "__main__":
    main()
