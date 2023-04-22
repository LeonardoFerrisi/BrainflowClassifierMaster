from brainflow import BoardShim, BoardIds
from brainflow import DataFilter, FilterTypes, AggOperations, DetrendOperations
import time
from termcolor import colored
import pygame


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


def collect_data(transpose=True, delay=10, iterations=10):
    import brainflow    
    from alive_progress import alive_bar

    pygame.init()
    pygame.mixer.init()

    board_id = select_board_id()

    # port = input(colored('Enter port: ', 'green'))
    
    params = brainflow.BrainFlowInputParams()
    
    params.serial_port = "COM3"
    print(f"BOARD ID: {board_id}")
    board = BoardShim(board_id, params)
    board.prepare_session()
    sound = pygame.mixer.Sound("start.mp3")
    sound2 = pygame.mixer.Sound("stop.mp3")
    
    input("Press Enter to continue...")

    dataset_x = []
    dataset_y = [1,0,1,0,1,0,1,0,1,0]

    for i in range(1, iterations):
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
        DataFilter.write_file(data, f'test_{i}.csv', 'w')  # use 'a' for append mode
        
        attempt = DataFilter.read_file(f'test_{i}.csv')
 
        
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

