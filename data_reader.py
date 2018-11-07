import numpy as np
import os
from setting import *


def extract_features(x):
    numrows = len(x)  # 3 rows in your example
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    var = np.var(x, axis=0)
    median = np.median(x, axis=0)
    xmax = np.amax(x, axis=0)
    xmin = np.amax(x, axis=0)
    p2p = xmax - xmin
    amp = xmax - mean
    s2e = x[numrows - 1, :] - x[0, :]
    features = np.concatenate([mean, std, var, median, xmax, xmin, p2p, amp, s2e])  # , morph]);
    return features


def data_saver(output_features, output_labels):
    np.save('all_sports_features.npy', output_features)
    np.save('all_sports_labels.npy', output_labels)


def data_generate_3(users_idx, acts_idx, segs_idx, address, rate):
    start = 0
    fsp = 0
    label = 0
    for u in users_idx:
        for s in segs_idx:
            for a in acts_idx:
                filename = address + acts[a] + '/' + 'p' + str(u) + '/' + segs[s] + '.txt'
                raw_data = np.genfromtxt(filename, delimiter=',', skip_header=0)
                for i in range(0, len(raw_data), rate):
                    f = np.zeros((1,0))
                    for location in range(5):
                        tmp = extract_features(raw_data[i:i + rate, location*9:(location+1) * 9]).reshape((1, -1))
                        f = np.hstack((f, tmp))
                    if start == 0:
                        fsp = f
                        label = a + 1
                        start = 1
                    else:
                        fsp = np.vstack((fsp, f))
                        label = np.vstack((label, a + 1))
    return fsp, label


def load_all_data_threaded(users_s, output_features, output_labels):
    segs_idx = range(0, 60)
    locations = ["T", "RA", "LA", "RL", "LL"]
    acts_idx = range(11, 19)
    for subject_name in users_s:
        features, labels = data_generate_3([subject_name], acts_idx, segs_idx, fn, rate=segment_rate)
        location_index = 0
        print('load data for subject ', subject_name)
        for sensor_location in locations:
            acc_features = features[:, location_index * 81:location_index * 81 + 27]
            gyro_features = features[:, location_index * 81 + 27:location_index * 81 + 54]
            magnet_features = features[:, location_index * 81 + 54:location_index * 81 + 81]
            output_features[users_s.index(subject_name), 0, location_index, :, :] = acc_features
            output_features[users_s.index(subject_name), 1, location_index, :, :] = gyro_features
            output_features[users_s.index(subject_name), 2, location_index, :, :] = magnet_features
            output_labels[users_s.index(subject_name), 0, location_index, :, :] = labels - 12
            output_labels[users_s.index(subject_name), 1, location_index, :, :] = labels - 12
            output_labels[users_s.index(subject_name), 2, location_index, :, :] = labels - 12
            location_index += 1
    return output_features, output_labels


def data_loader_file():
    if os.path.isfile('all_sports_features.npy'):
        all_features = np.load('all_sports_features.npy')
        all_labels = np.load('all_sports_labels.npy')
        return all_features, all_labels
    else:
        return -1, -1


def load_all_data(user_input, input_features, input_labels):
    output_features, output_labels = data_loader_file()
    if type(output_labels) == int:
        output_features, output_labels = load_all_data_threaded(user_input, input_features, input_labels)
        print("data has been loaded")
        data_saver(output_features, output_labels)
    return output_features, output_labels
