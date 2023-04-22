from brainflow import BoardShim, BoardIds
from brainflow import DataFilter, FilterTypes, AggOperations, DetrendOperations
import time
from termcolor import colored
import pygame
import os

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


def collect_data(transpose=True, delay=30, iterations=10):
    """
    Collects data for processing with Machine Learning from EEG
    """

    import brainflow    
    from alive_progress import alive_bar

    pygame.init()
    pygame.mixer.init()

    board_id = select_board_id()

    params = brainflow.BrainFlowInputParams()

    config = parse_config("config.dat")
    if board_id != brainflow.BoardIds.UNICORN_BOARD.value:
        port = config["PORT"]
        new_config = input(f"Use new serial port, or used stored serial port? Stored: {port}")
        if new_config != "":
            port = new_config
        params.serial_port = port
    # port = input(colored('Enter port: ', 'green'))
    
    print(f"BOARD ID: {board_id}")
    board = BoardShim(board_id, params)
    board.prepare_session()
    sound = pygame.mixer.Sound("start.mp3")
    sound2 = pygame.mixer.Sound("stop.mp3")
    
    input("Press Enter to continue...")

    dataset_y_config = input("""
    Please select a configuration for your dataset format from the following:

    1. [ON, OFF, ON, OFF, ON, OFF, ON, OFF, ON, OFF]
    2. [OFF, ON, OFF, ON, OFF, ON, OFF, ON, OFF, ON]
    3. [ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON]
    4. [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF]                         
    """)

    dy_params = {
        "1": [1,0,1,0,1,0,1,0,1,0],
        "2": [0,1,0,1,0,1,0,1,0,0],
        "3": [1,1,1,1,1,1,1,1,1,1],
        "4": [0,0,0,0,0,0,0,0,0,0]
    }
    if dataset_y_config in list(dy_params.keys()):
        dataset_y = dy_params[dataset_y_config]

    dataset_x = []

    for i in range(1, iterations+1):
        print(f"\n\nRunning Session {i}...\n\n")
        sound.play()
        time.sleep(2)

        board.start_stream()

        # time.sleep(10)

        # use alive_bar to display progress
        with alive_bar(delay) as bar:
            for n in range(delay):
                time.sleep(1)
                bar() # increment progress bar

        # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
        data = board.get_board_data()  # get all data and remove it from internal buffer
        board.stop_stream()
        
        import numpy as np
        import os
        
        datapath = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(datapath):
            os.makedirs(datapath)

        current_value = dataset_y[i-1]
        filename = f'test_{i}_{current_value}.csv'

        filepath = os.path.join(datapath, filename)
        DataFilter.write_file(data, filepath, 'w')  # use 'a' for append mode
        
        attempt = DataFilter.read_file(filepath)
        
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
        dataset_x.append(bands[0])
        
        with open(f'vector_{i}.txt', "w") as f:
            f.write(str(bands))

        data = None
        sound2.play()
        time.sleep(2)
    
    board.release_session()
    return dataset_x, dataset_y

if __name__ == "__main__":
    dataset_x, dataset_y = collect_data()
    print("Data collection complete.")
