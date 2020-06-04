import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from obspy.clients.iris import Client
import numpy as np
import pickle
import multiprocessing as mp
import os

NOISE_N = 1  # number of noise windows to extract per actual pick

CLIENT = Client()
NETWORK = "RV"
CHANNELS = ["HHN", "HHE", "HHZ"]
W = 4  # 4 s window
NORMALIZE=True

def plot_window(x_df, category):
    fig, axes = plt.subplots(nrows=3, sharex=True)
    for i, c in enumerate(CHANNELS):
        axes[i].plot(x_df[c], label=c)

        axes[i].set_ylabel("Amplitude")
        axes[i].legend()
    plt.xlabel("Time")
    plt.suptitle(category)
    plt.show()
    plt.clf()


def extract_window(station, t1, t2):
    x_df = pd.DataFrame()
    for c in CHANNELS:
        stream = CLIENT.timeseries(network=NETWORK, station=station, location='--', channel=c, starttime=t1, endtime=t2)
        x_df[c] = stream[0].data
        # normalize window
        if NORMALIZE:
            x_df[c] = (x_df[c] - np.nanmean(x_df[c])) / np.nanstd(x_df[c])
    x_df = x_df[CHANNELS]

    return x_df



def extract_pick(station, y, time, x_list, y_list, plot):
    x_df = None
    for i in range(0, 100):
        try:
            x_df = extract_window(station, time - W/2, time + W/2)
            break
        except:  # when data can't be found on network
            continue

    if x_df is None:
        print("Could not reach network")
        return x_list, y_list

    if plot is True:
        plot_window(x_df, y)

    x_list.append(x_df)
    y_list.append(y)
    return x_list, y_list


def extract_noise(t1, t2, station, x_list, y_list, plot):

    if t2 - W/2 < t1 + W/2:
        return x_list, y_list
    random_samples = np.arange(start=int(t1+W/2), stop=int(t2-W/2), step=1)
    if len(random_samples) > NOISE_N:
        random_samples = list(set(np.random.choice(random_samples, size=NOISE_N, replace=False)))
        for i, random_time in enumerate(random_samples):
            try:
                x_df = extract_window(station, random_time - W/2, random_time + W/2)
            except:
                print("Could not reach network")
                continue
            if plot is True and i == 0:

                plot_window(x_df, "Noise")
                break
            x_list.append(x_df)
            y_list.append('Noise')

    return x_list, y_list


def get_labeled_data(picks, p, plot):
    for station in picks["sta"].unique():

        sub_picks = picks[picks["sta"] == station]
        sub_picks.sort_values(by="time", inplace=True)
        x_list = list()
        y_list = list()
        for i, time in enumerate(tqdm(sub_picks["time"], desc="Extracting station = %s, %i" % (station, p))):
            y = sub_picks["iphase"].iloc[i]
            # extract pick window
            x_list, y_list = extract_pick(station, y, time, x_list, y_list, plot=plot)

            # extract noise windows
            if i > len(sub_picks.index) - 2:
                continue
            t2 = sub_picks["time"].iloc[i+1]
            x_list, y_list = extract_noise(time, t2, station, x_list, y_list, plot=plot)

        pickle.dump({"x": x_list, "y": y_list}, open("data/%s_labeled.pkl" % station, "wb"), protocol=4)


def main(pick_loc="data/Picks.xlsx", plot=False, n_jobs=-1):

    if not os.path.isdir("data"):
        os.mkdir("data")
        os.mkdir("data")
    picks = pd.read_excel(pick_loc)

    if n_jobs == -1:
        n_jobs = os.cpu_count() - 1

    stations = list(picks["sta"].unique())
    chunks = np.array_split(stations, n_jobs)

    processes = list()
    for i, c in enumerate(chunks):
        sub_picks = picks[picks["sta"].isin(c)]
        p = mp.Process(target=get_labeled_data, args=(sub_picks, i, plot))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()


