%%writefile create_dataset.py
import argparse
import pandas as pd
import os
from scipy import signal
import pickle # Import pickle module

def load_data(file_path):
    """Loads physiological data from a file into a pandas DataFrame."""
    df = pd.read_csv(file_path, skiprows=7, delimiter=';', decimal=',', names=['Timestamp', 'Value'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M:%S,%f')
    df = df.set_index('Timestamp')
    return df

def parse_event_duration(row_raw):
    """Parses event duration strings into start and end datetime objects."""
    parts = row_raw.split('-')
    start_str = parts[0].strip()
    end_str = parts[1].strip()

    start_dt = pd.to_datetime(start_str, format='%d.%m.%Y %H:%M:%S,%f')

    # If end time doesn't contain a date, assume it's the same day as start date
    if '.' not in end_str:
        end_str = start_dt.strftime('%d.%m.%Y ') + end_str

    end_dt = pd.to_datetime(end_str, format='%d.%m.%Y %H:%M:%S,%f')

    return start_dt, end_dt

def apply_filter(data, signal_column, sampling_rate):
    """Applies a bandpass filter to the specified signal column."""
    lowcut = 0.17
    highcut = 0.4
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    order = 2
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, data[signal_column].values)
    print(f"Applied bandpass filter to {signal_column} with cutoffs {lowcut}-{highcut} Hz.")
    return filtered_signal

def main():
    parser = argparse.ArgumentParser(description='Process physiological data and events to create a dataset.')
    parser.add_argument('--input_directory', type=str, required=True,
                        help='Path to the directory containing input physiological data files (Flow, Thorac, SPO2, Flow Events).')
    parser.add_argument('--output_directory', type=str, required=True,
                        help='Path to the directory where the processed dataset will be saved.')

    args = parser.parse_args()

    print(f"Input Directory: {args.input_directory}")
    print(f"Output Directory: {args.output_directory}")

    # file paths
    flow_file_path = os.path.join(args.input_directory, 'Flow  - 30.05.2024.txt')
    thorac_file_path = os.path.join(args.input_directory, 'Thorac  - 30.05.2024.txt')
    spo2_file_path = os.path.join(args.input_directory, 'SPO2  - 30.05.2024.txt')
    event_file_path = os.path.join(args.input_directory, 'Flow Events  - 30.05.2024.txt')

    # physiological data
    df_flow = load_data(flow_file_path)
    df_thorac = load_data(thorac_file_path)
    df_spo2 = load_data(spo2_file_path)

    #event data
    df_events = pd.read_csv(event_file_path, skiprows=6, delimiter=';', names=['Event_Duration_Raw', 'Duration_in_Seconds', 'Event_Type', 'Sleep_Stage'])
    df_events[['Start_Time', 'End_Time']] = df_events['Event_Duration_Raw'].apply(lambda x: pd.Series(parse_event_duration(x)))

    # value renaming
    df_flow = df_flow.rename(columns={'Value': 'Flow'})
    df_thorac = df_thorac.rename(columns={'Value': 'Thorac'})
    df_spo2 = df_spo2.rename(columns={'Value': 'SPO2'})

    # combining
    df_respiration_signals = pd.merge(df_flow, df_thorac, left_index=True, right_index=True, how='outer')
    df_all_signals = pd.merge(df_respiration_signals, df_spo2, left_index=True, right_index=True, how='outer')
    df_all_signals['SPO2'] = df_all_signals['SPO2'].interpolate(method='linear')

    # bandpass filter to Flow and Thorac signals
    sampling_rate_respiration = 32
    df_all_signals['Flow_Filtered'] = apply_filter(df_all_signals, 'Flow', sampling_rate_respiration)
    df_all_signals['Thorac_Filtered'] = apply_filter(df_all_signals, 'Thorac', sampling_rate_respiration)

    print("\ndf_all_signals head after filtering:")
    print(df_all_signals.head())
    print("\ndf_all_signals info after filtering:")
    df_all_signals.info()

    print("\ndf_events head:")
    print(df_events.head())
    print("\ndf_events info:")
    df_events.info()

    # Windowing and labeling logic
    window_size = pd.Timedelta(seconds=30)
    overlap_percentage = 0.5
    step_size = window_size * (1 - overlap_percentage)

    processed_windows = []

    current_time = df_all_signals.index.min()
    end_of_signals = df_all_signals.index.max()

    window_counter = 0
    while current_time + window_size <= end_of_signals:
        window_start = current_time
        window_end = current_time + window_size
        df_window = df_all_signals.loc[window_start:window_end]

        if not df_window.empty:
            assigned_label = 'Normal'
            max_overlap_percentage = 0.0

            for _, event_row in df_events.iterrows():
                event_start = event_row['Start_Time']
                event_end = event_row['End_Time']

                # overlap between current window and event
                overlap_start = max(window_start, event_start)
                overlap_end = min(window_end, event_end)

                if overlap_end > overlap_start:
                    overlap_duration = (overlap_end - overlap_start).total_seconds()
                    window_duration_sec = window_size.total_seconds()
                    overlap_percent = (overlap_duration / window_duration_sec) * 100

                    if overlap_percent > 50 and overlap_percent > max_overlap_percentage:
                        assigned_label = event_row['Event_Type']
                        max_overlap_percentage = overlap_percent

            processed_windows.append({
                'data': df_window,
                'label': assigned_label,
                'start_time': window_start,
                'end_time': window_end
            })

        current_time += step_size
        window_counter += 1

    print(f"\nTotal windows created: {len(processed_windows)}")
    if processed_windows:
        print("First 3 processed windows examples:")
        for i in range(min(3, len(processed_windows))):
            window_data = processed_windows[i]
            print(f"  Window {i+1}:")
            print(f"    Start: {window_data['start_time']}, End: {window_data['end_time']}")
            print(f"    Label: {window_data['label']}")
            print(f"    Data Shape: {window_data['data'].shape}")

    # if output directory doesn't exist
    os.makedirs(args.output_directory, exist_ok=True)

    # Define output file path for the pickled dataset
    output_file_path = os.path.join(args.output_directory, 'processed_dataset.pkl')

    # Save the processed_windows list to a pickle file
    with open(output_file_path, 'wb') as f:
        pickle.dump(processed_windows, f)

    print(f"\nProcessed dataset saved to: {output_file_path}")

if __name__ == '__main__':
    main()
