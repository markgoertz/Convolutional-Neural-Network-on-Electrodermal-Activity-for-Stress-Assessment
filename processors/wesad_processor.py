import os
import logging
from utilities.io_utils import unpickle_file, save_dataframe_to_csv, save_windows_as_pickle
import pandas as pd
from utilities.data_utils import combine_windows_into_dataframe
from processors.label_processor import LabelProcessor
from processors.signal_processor import SignalProcessor
from processors.window_processor import WindowProcessor
from tqdm import tqdm

class WESADProcessor:
    """
    A processor class for handling WESAD dataset processing.
    Attributes:
        config (dict): Configuration dictionary for processing parameters.
        logger (logging.Logger): Logger instance for logging errors and warnings.
    """

    def __init__(self, config):
        """
        Initializes the WESADProcessor with the given configuration.
        Args:
            config (dict): Configuration dictionary for processing parameters.
        """
        self.config = config
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def process_subject(self, subject_path, subject_id):
        """
        Processes all files for a given subject.
        Args:
            subject_path (str): Path to the subject's data directory.
            subject_id (str): Identifier for the subject.
        Raises:
            Exception: If an error occurs during processing.
        """
        try:
            files = list(filter(lambda f: f.endswith('.pkl'), os.listdir(subject_path)))
            for file in tqdm(files, desc=f"Processing files for subject {subject_id}"):
                self.process_file(subject_path, file, subject_id)
        except Exception as e:
            self.logger.error(f"Error processing subject {subject_id} at {subject_path}: {e}", exc_info=True)

    def process_file(self, subject_path, file, subject_id):
        """
        Processes a single file for a given subject.
        Args:
            subject_path (str): Path to the subject's data directory.
            file (str): Filename of the data file to process.
            subject_id (str): Identifier for the subject.
        Raises:
            Exception: If an error occurs during processing.
        """
        try:
            data = unpickle_file(os.path.join(subject_path, file))
            if not self.is_valid_data(data):
                self.logger.warning(f"Invalid data format in file {file} for subject {subject_id}")
                return

            signals, labels = data['signal']['wrist'], data['label']
                        
            labels = self.process_labels(labels)
            signals = self.process_signals(signals)
            
            windowed_signals, majority_labels = self.process_windows(signals, labels)
            
            df_windows, session_name = combine_windows_into_dataframe(
                windowed_signals, majority_labels, subject_id, self.config['preprocessed'], self.config['window_duration_seconds'], self.config['timestamp'], self.config['metrics']
            )
           
            save_dataframe_to_csv(df_windows, subject_id, session_name)
            save_windows_as_pickle(
                windowed_signals = windowed_signals,
                metrics = list(self.config["target_rate"].keys()),
                labels = majority_labels,
                subject_id = subject_id,
                session_name = 'unknown',
                base_path= os.path.join(os.getcwd(), 'data'),
                start_time=self.config['timestamp'],
                window_duration_seconds=self.config['window_duration_seconds']
            )            
        except Exception as e:
            self.logger.error(f"Error processing file {file} for subject {subject_id} at {subject_path}: {e}", exc_info=True)

    def is_valid_data(self, data):
        """
        Validates the data format.
        Args:
            data (dict): Data dictionary to validate.
        Returns:
            bool: True if data is valid, False otherwise.
        """
        return data and 'signal' in data and 'label' in data

    def process_labels(self, labels):
        """
        Processes the labels by downsampling and encoding them.
        Args:
            labels (list): List of labels to process.
        Returns:
            list: Processed labels.
        """
        label_proc = LabelProcessor(700, 4)
        return label_proc.encode(label_proc.downsample(labels))

    def process_signals(self, signals):
        """
        Processes the signals by extracting, normalizing, and downsampling them.
        Args:
            signals (dict): Dictionary of signals to process.
        Returns:
            dict: Processed signals.
        """
        signals = SignalProcessor.extract_all_signals(signals)
        signals = {k: SignalProcessor.normalize(v) for k, v in signals.items()}
        signals['BVP'] = SignalProcessor.downsample(signals['BVP'], 64, 32)
        return signals

    def process_windows(self, signals, labels):
        """
        Processes the signals and labels into windows.
        Args:
            signals (dict): Dictionary of signals to process.
            labels (list): List of labels to process.
        Returns:
            tuple: Tuple containing windowed signals and majority labels.
        """
        window_sizes = {k: self.config['target_rate'][k] * self.config['window_duration_seconds'] for k in signals.keys()}       
        windowed_signals = WindowProcessor.apply_to_signals(signals, window_sizes)
        labels_windows = WindowProcessor.sliding_window(labels, 32, 32)
        majority_labels = [LabelProcessor.majority_vote(w) for w in labels_windows]

        return windowed_signals, majority_labels

    def process_dataset(self, base_path):
        """
        Processes the entire dataset by processing each subject.
        Args:
            base_path (str): Path to the base directory of the dataset.
        Raises:
            Exception: If an error occurs during processing.
        """
        try:
            subjects = list(filter(lambda x: os.path.isdir(os.path.join(base_path, x)), os.listdir(base_path)))
            for subject in tqdm(subjects, desc="Processing subjects"):
                self.process_subject(os.path.join(base_path, subject), subject)
        except Exception as e:
            self.logger.error(f"Error processing dataset at {base_path}: {e}", exc_info=True)
