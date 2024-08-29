import pandas as pd
import tqdm
import wfdb
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dataframe(subject):
    # print(f'Processing subject: {subject}...')
    '''
    input: subject_id (int)
    output: path_list (list of dataframes)
    '''
    # read record_list.csv
    record_list = pd.read_csv('record_list.csv')
    path_list = record_list[record_list['subject_id'] == subject]['path'].to_list()



    ############# read every available record from each patient #############

    df = []
    for path in path_list:
        try:
            rd_record = wfdb.rdrecord(path)
            df.append(pd.DataFrame(rd_record.p_signal, columns=rd_record.sig_name))
            print(f'Processing subject: {subject}...')
        except Exception as e:
            # logging.error(f"Error processing subject: {subject}, path: {path}, error: {str(e)}")

            continue
        
    ############# read only the first available record from each patient #############

    # df = []
    # while len(df) == 0:
    #     try:

    #         re_record = wfdb.rdrecord(path_list[0])
    #         df.append(pd.DataFrame(re_record.p_signal, columns=re_record.sig_name))
    #         print(f'Processing subject: {subject}...')
    #     except:
    #         # print('Error reading record, trying next record...')
    #         break

    return df

def is_peak(arr, i):
    """
    input: arr (list), i (int)
    output: boolean
    """
    if i == 0 or i == len(arr) - 1:
        return False
    if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
        return True
    return False

def active_segment(df):
    # first calculate the mean of the signal
    mean = df['I'].mean()
    # then calculate the standard deviation of the signal
    std = df['I'].std()
    # then calculate the threshold
    threshold = mean + 3 * std
    # then calculate the active segment
    active_segment = []
    i = 0
    while i < len(df):
        # find peak that is greater than threshold
        if df['I'][i] > threshold and is_peak(df['I'], i):
            start = i - 100
            end = i + 100
            if start < 0:
                start = 0
            if end >= len(df):
                end = len(df) - 1

            tmp = df[start:end]
    
            if len(tmp) == 200:
                active_segment.append(tmp)
            i += 100
        else:
            i += 1
            
    return active_segment

def save_hdf5(dataset, label):
    '''
    input: dataset (list of dataframes), label (int)
    output: None
    '''
    dir_path = 'processed_data'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    hdf5_path = os.path.join(dir_path, f'{label}.h5')
    with pd.HDFStore(hdf5_path, 'w') as store:
        for i, data in enumerate(dataset):
            store.put(f'data_{i}', data, format='table')

    logging.info('Done!')

def normalize(df):
    """
    input: df (dataframe)
    output: df (dataframe)
    """
    # for each column, normalize the data with min-max normalization
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
    return df

def create_dataset(patient, label):
    '''
    input: patient (list of int), segment_length (int), label (int)
    output: dataset (dataframe)
    '''
    # find the correspond ecg data to the subject ids
    df = []
    logging.info(f'Finding the corresponding ECG data for class: {label}...')
    for subject in tqdm.tqdm(patient):
        for record in get_dataframe(subject):
            record = normalize(record)
            df.append(record)
            
    save_hdf5(df, label)
    return 0

def segment_data(data, segment_length):
    """
    Segments the input data into chunks of specified segment_length.

    Parameters:
    data (numpy.ndarray): Input data of shape (samples, timesteps, features).
    segment_length (int): Length of each segment.

    Returns:
    numpy.ndarray: Segmented data of shape (new_samples, segment_length, features).
    """
    data = np.array(data)
    samples, timesteps, features = data.shape
    if timesteps % segment_length != 0:
        raise ValueError(f"The timesteps dimension must be divisible by {segment_length}.")

    # Calculate the number of segments per sample
    num_segments = timesteps // segment_length

    # Initialize a list to collect segments
    segmented_data = []

    # Iterate over each sample and segment it
    for sample in tqdm.tqdm(range(samples)):
        for segment in range(num_segments):
            start_idx = segment * segment_length
            end_idx = start_idx + segment_length
            segmented_data.append(data[sample, start_idx:end_idx, :])

    # Convert the list to a numpy array
    segmented_data = np.array(segmented_data)
    return segmented_data

def load_dataset(label):
    '''
    input: label (int)
    output: dataset (list of dataframes)
    '''
    # load the dataset
    dataset = []
    hdf5_path = os.path.join('processed_data', f'{label}.h5')
    with pd.HDFStore(hdf5_path, 'r') as store:
        keys = store.keys()
        for key in tqdm.tqdm(keys):

            dataset.append(store[key])
    
    return dataset



def segment_array(arr, segment_length):
    """
    Segments each M-length segment of the input array into five L-length segments along axis 2.

    Parameters:
    - arr: numpy array of shape (N, 12, M)
    - segment_length: length L of each segment to split into

    Returns:
    - segmented_array: numpy array of shape (N * 5, 12, M // segment_length)
    """
    assert arr.shape[2] % segment_length == 0, \
        "Segment length must divide the size of axis 2 of the input array evenly."

    # Reshape to facilitate splitting
    num_segments = arr.shape[2] // segment_length
    array_reshaped = arr.reshape(arr.shape[0], arr.shape[1], num_segments, segment_length)

    # Split along axis 2
    segments = np.split(array_reshaped, num_segments, axis=2)

    # Concatenate and reshape back to the desired shape
    segmented_array = np.concatenate(segments, axis=0).reshape(-1, arr.shape[1], segment_length)

    return segmented_array





# Example usage
if __name__ == "__main__":
    subjects = [1, 2, 3]  # Example subject IDs
    segment_length = 100  # Example segment length
    label = 'example_label'  # Example label

    create_dataset(subjects, segment_length, label)


