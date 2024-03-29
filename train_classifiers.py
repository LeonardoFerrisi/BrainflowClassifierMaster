import glob
import argparse
import os
import pickle
import logging

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from termcolor import colored


def write_model(intercept, coefs, model_type):
    coefficients_string = '%s' % (','.join([str(x) for x in coefs[0]]))
    file_content = '''
#include "%s"
// clang-format off
const double %s_coefficients[%d] = {%s};
double %s_intercept = %lf;
// clang-format on
''' % (f'{model_type}_model.h', model_type, len(coefs[0]), coefficients_string, model_type, intercept)
    file_name = f'{model_type}_model.cpp'
    # file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'generated', file_name)
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'generated', file_name)

    with open(file_path, 'w') as f:
        f.write(file_content)

def prepare_data(first_class, second_class, board_id, blacklisted_channels=None):
    # use different windows, its kinda data augmentation
    window_sizes = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    overlaps = [0.5, 0.475, 0.45, 0.425, 0.4, 0.375, 0.35] # percentage of window_size
    dataset_x = list()
    dataset_y = list()
    print("attempting to pickle...")
    for data_type in (first_class, second_class):
        print(f"attempting to read files... in {os.path.join('data', data_type)}")
        dir = os.path.join('data', data_type)
        for file in os.listdir(dir):
            print(f"Found file: {file}")
            logging.info(file)
            # board_id = os.path.basename(os.path.dirname(file))
            try:
                board_id = int(board_id)
                filepath = os.path.join('data', data_type, file)
                print(f"Attmepting to read file: {filepath}")
                data = DataFilter.read_file(filepath)
                sampling_rate = BoardShim.get_sampling_rate(board_id)
                eeg_channels = get_eeg_channels(board_id, blacklisted_channels)
                print(f"Sampling rate: {sampling_rate}")
                print(f"EEG Channels: {eeg_channels}")
                for num, window_size in enumerate(window_sizes):
                    # cur_pos = sampling_rate * 10
                    cur_pos = 0
                    while cur_pos + int(window_size * sampling_rate) < data.shape[1]:
                        data_in_window = data[:, cur_pos:cur_pos + int(window_size * sampling_rate)]
                        data_in_window = np.ascontiguousarray(data_in_window)
                        bands = DataFilter.get_avg_band_powers(data_in_window, eeg_channels, sampling_rate, True)
                        feature_vector = bands[0]
                        feature_vector = feature_vector.astype(float)
                        dataset_x.append(feature_vector)
                        if data_type == first_class:
                            dataset_y.append(0)
                        else:
                            dataset_y.append(1)
                        cur_pos = cur_pos + int(window_size * overlaps[num] * sampling_rate)
            except Exception as e:
                logging.error(str(e), exc_info=True)

    print("Dataset_x: ", dataset_x)
    print("Dataset_y: ", dataset_y)
    print("\n=====================\n")
    logging.info('1st Class: %d 2nd Class: %d' % (len([x for x in dataset_y if x == 0]), len([x for x in dataset_y if x == 1])))

    with open('dataset_x.pickle', 'wb') as f:
        pickle.dump(dataset_x, f, protocol=3)
    with open('dataset_y.pickle', 'wb') as f:
        pickle.dump(dataset_y, f, protocol=3)

    return dataset_x, dataset_y

def get_eeg_channels(board_id, blacklisted_channels):
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    try:
        eeg_names = BoardShim.get_eeg_names(board_id)
        selected_channels = list()
        if blacklisted_channels is None:
            blacklisted_channels = set()
        for i, channel in enumerate(eeg_names):
            if not channel in blacklisted_channels:
                selected_channels.append(eeg_channels[i])
        eeg_channels = selected_channels
    except Exception as e:
        logging.warn(str(e))
    logging.info('channels to use: %s' % str(eeg_channels))
    return eeg_channels

def print_dataset_info(data):
    x, y = data
    first_class_ids = [idx[0] for idx in enumerate(y) if idx[1] == 0]
    second_class_ids = [idx[0] for idx in enumerate(y) if idx[1] == 1]
    x_first_class = list()
    x_second_class = list()
    
    for i, x_data in enumerate(x):
        if i in first_class_ids:
            x_first_class.append(x_data.tolist())
        elif i in second_class_ids:
            x_second_class.append(x_data.tolist())
    second_class_np = np.array(x_second_class)
    first_class_np = np.array(x_first_class)

    logging.info('1st Class Dataset Info:')
    logging.info('Mean:')
    logging.info(np.mean(first_class_np, axis=0))
    logging.info('2nd Class Dataset Info:')
    logging.info('Mean:')
    logging.info(np.mean(second_class_np, axis=0))

def train_regression(data, metric:str='neural'):
    model = LogisticRegression(solver='liblinear', max_iter=4000,
                                penalty='l2', random_state=2, fit_intercept=True, intercept_scaling=0.2)
    logging.info('#### Logistic Regression ####')
    scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=8)
    logging.info('f1 macro %s' % str(scores))
    model.fit(data[0], data[1])
    logging.info(model.intercept_)
    logging.info(model.coef_)
    
    initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
    with open('logreg_model.onnx', 'wb') as f:
        f.write(onx.SerializeToString())
    write_model(model.intercept_, model.coef_, f'{metric}')

def train_svm(data, metric:str='neural'):
    model = SVC(kernel='linear', verbose=True, random_state=1, class_weight='balanced', probability=True)
    logging.info('#### SVM ####')
    model.fit(data[0], data[1])
    initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
    with open('svm_model.onnx', 'wb') as f:
        f.write(onx.SerializeToString())

def train_random_forest(data, metric:str='neural'):
    model = RandomForestClassifier(class_weight='balanced', random_state=1, n_jobs=15, n_estimators=200)
    logging.info('#### Random Forest ####')
    scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=15)
    logging.info('f1 macro %s' % str(scores))
    model.fit(data[0], data[1])

    initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
    with open('forest_model.onnx', 'wb') as f:
        f.write(onx.SerializeToString())

def train_knn(data, metric:str='neural'):
    model = KNeighborsClassifier(n_neighbors=10, n_jobs=8)
    logging.info('#### KNN ####')
    scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=15)
    logging.info('f1 macro %s' % str(scores))
    model.fit(data[0], data[1])

    initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
    with open('knn_model.onnx', 'wb') as f:
        f.write(onx.SerializeToString())

def train_mlp(data, metric:str='neural'):
    model = MLPClassifier(hidden_layer_sizes=(100, 20),learning_rate='adaptive', max_iter=1000,
                          random_state=1, verbose=True, activation='logistic', solver='adam')
    logging.info('#### MLP ####')
    scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=15)
    logging.info('f1 macro %s' % str(scores))
    model.fit(data[0], data[1])

    initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
    with open('mlp_model.onnx', 'wb') as f:
        f.write(onx.SerializeToString())

def train_stacking_classifier(data, metric:str='neural'):
    model1 = MLPClassifier(hidden_layer_sizes=(100, 20),learning_rate='adaptive', max_iter=1000,
                          random_state=1, verbose=True, activation='logistic', solver='adam')
    model2 = KNeighborsClassifier(n_neighbors=10, n_jobs=8)
    model3 = RandomForestClassifier(class_weight='balanced', random_state=1, n_jobs=8, n_estimators=200)
    meta_model = LogisticRegression()
    sclf = StackingClassifier(estimators=[('MLPClassifier', model1), ('KNeighborsClassifier', model2), ('RandomForestClassifier', model3)],
                              final_estimator=meta_model, n_jobs=15,
                              passthrough=True)
    logging.info('#### Stacking ####')
    scores = cross_val_score(sclf, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=15)
    logging.info('f1 macro %s' % str(scores))
    sclf.fit(data[0], data[1])

    initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
    onx = convert_sklearn(sclf, initial_types=initial_type, target_opset=11, options={type(sclf): {'zipmap': False}})
    with open('stacking_model.onnx', 'wb') as f:
        f.write(onx.SerializeToString())

def main(reuse=True, board_id=-1):
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-dataset', action='store_true')
    args = parser.parse_args()

    metric = input("Please enter a label for your metric: ")

    if args.reuse_dataset or reuse:
        with open('dataset_x.pickle', 'rb') as f:
            dataset_x = pickle.load(f)
        with open('dataset_y.pickle', 'rb') as f:
            dataset_y = pickle.load(f)
        data = dataset_x, dataset_y
    else:
        # data = prepare_data('relaxed', 'focused', blacklisted_channels={'T3', 'T4'})
        print("Preparing data...")
        data = prepare_data(f'{metric}', f'not_{metric}', blacklisted_channels={'T3', 'T4'}, board_id=board_id)

    

    print("DATA prepared")

    print_dataset_info(data, metric)
    train_regression(data, metric)
    train_svm(data, metric)
    train_knn(data, metric)
    train_random_forest(data, metric)
    train_mlp(data, metric)
    train_stacking_classifier(data, metric)

def select_board_id():
    """
    prompts user to select a board id from a list of available devices
    """
    from brainflow import BoardShim, BoardIds

    board_prompt = """
    ----------------
    1: Muse 2
    2: Cyton
    3: Ganglion
    4: Muse 2016
    5: Gtec Unicorn
    ----------------
    """
    print(board_prompt)
    user_select = input(colored('Select Board ID: ', 'green'))

    id_pairs = {
        "1": BoardIds.MUSE_2_BLED_BOARD.value,
        "2": BoardIds.CYTON_BOARD.value,
        "3": BoardIds.GANGLION_BOARD.value,
        "4": BoardIds.MUSE_2016_BLED_BOARD.value,
        "5": BoardIds.UNICORN_BOARD.value
    }
    if user_select in list(id_pairs.keys()):
        print(id_pairs[user_select])
        return id_pairs[user_select]
    else:
        return None
    
if __name__ == '__main__':
    board_id = select_board_id()
    main(reuse=False, board_id=board_id)
