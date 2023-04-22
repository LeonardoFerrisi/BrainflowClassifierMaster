import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
import os
from termcolor import colored

def parse_config(config_file):
    """
    Checks the local config file and returns it's contents as a dictionary
    """

    dir = os.getcwd()
    config_path = os.path.join(dir, config_file)

    config_dict = {}

    with open(config_path, "r") as f:
        for line in f:
            
            content = line.split(": ")
            print(content)
            config_dict[content[0]] = content[1]

    return config_dict

def select_board_id():
    """
    prompts user to select a board id from a list of available devices
    """
    
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

def print_dict(d:dict):
    print("------------------------")
    for key, value in d.items():
        print(f"    {key}: {value}")

def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    params = BrainFlowInputParams()

    board_id = select_board_id()

    board = BoardShim(board_id, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    board.prepare_session()

    print("\n\n=================\n\n")
    print("Load ML model:")
    models_available = os.listdir(os.path.join(os.getcwd(), 'models'))
    model_dict = {}
    for i, model in enumerate(models_available):
        model_dict[str(i)] = model
    print_dict(model_dict)

    model_select = input("Which model would you like to select?")
    
    current_model = None
    if model_select in list(model_dict.keys()):
        current_model = model_dict[int(model_select)]

    model_params = BrainFlowModelParams(BrainFlowMetrics.USER_DEFINED.value,
                                              BrainFlowClassifiers.ONNX_CLASSIFIER)
    model_params.file = os.path.join(os.getcwd(), 'model', current_model)
    model_params.output_name = "probabilities"

    ml_model = MLModel(model_params)
    ml_model.prepare()

    input("Start Stream?")
    print("At any time, click any key 'x' to stop streaming.")

    board.start_stream(45000)
    
    keep_running = True
    while keep_running:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
        time.sleep(5)  # recommended window size for eeg metric calculation is at least 4 seconds, bigger is better
        data = board.get_board_data()
        eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
        bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
        feature_vector = bands[0]    
        ml_model.predict(feature_vector)
    
    ml_model.release()
    board.stop_stream()
    board.release_session()


if __name__ == "__main__":
    main()