import torch
import torch.nn as nn
import scipy.io as sio
import glob
import numpy as np
import mat73
import resampy
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Tuple, Optional
import random
from scipy.signal import butter, filtfilt, iirnotch
import portion as P

"""
1. Normalization: 
    >9000 | <9000 -> np.nan  # TODO: should do interpolation instead of replaced by mean
    mean -> np.nanmean(data)
    np.nan -> mean
    standard                # TODO: should be done after train/test split, and only calculate on train and apply on train & test
2. Bandpass Filter
3. Notch Filter: 50 & 60 Hz
4. Downsample: 1024 Hz -> 256 Hz
5. Segment as 2 sec
6. Label and remove segments overlapped with nan_indices
6. On Train: class balance
2025-04-14
"""


# Constants
TARGET_FS = 256  # Target sampling frequency
WINDOW_SIZE_SEC = 2  # 2-second segments
OVERLAP_SEC = 0  # No overlap between segments
CHANNEL_DICT = {
    'LEAR1': 'LEAR1.mat',
    'LEAR2': 'LEAR2.mat',
    'REAR1': 'REAR1.mat',
    'REAR2': 'REAR2.mat'
}

# random seeds
global_seed = 2024
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)

def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    """Design Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int) -> np.ndarray:
    """Apply bandpass filter to data."""
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def apply_notch_filter(data: np.ndarray, freq: float, Q: float, fs: int) -> np.ndarray:
    """Apply notch filter to remove specific frequency."""
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

def remove_empty_intervals(intervals: P.Interval) -> P.Interval:
    """Remove empty intervals where end <= start."""
    clean_intervals = P.empty()
    for interval in intervals:
        if interval.upper > interval.lower:     # valid interval
            clean_intervals |= interval
    return clean_intervals

def merge_overlapping_intervals(intervals: P.Interval) -> P.Interval:
    """Merge overlapping intervals."""
    if intervals.empty:
        return P.empty()

    merged_intervals = P.empty()
    sorted_intervals = sorted(intervals, key=lambda x: x.lower)
    current = sorted_intervals[0]
    for next_interval in sorted_intervals[1:]:
        if current.overlaps(next_interval) or current.upper == next_interval.lower:
            current = P.closed(current.lower, max(current.upper, next_interval.upper))
        else:
            merged_intervals |= current
            current = next_interval

    merged_intervals |= current
    return merged_intervals

class SeizureDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, window_size: int, fs: int, balance: bool = False, seed: int = 42):
        """
        Args:
            data: numpy array of shape (num_channels, num_samples)
            labels: numpy array of shape (num_samples,)
            window_size: window size in samples
            fs: sampling frequency
        """
        self.fs = fs
        self.window_size = window_size
    

        # Apply preprocessing
        self.data = data
        self.labels = labels
        
        # Segment data into window_size
        segments_data, segments_labels = self.segment_data()

        if balance:
            np.random.seed(global_seed)
            segments_data, segments_labels = self.balance_classes(segments_data, segments_labels)
        
        self.segments_data = segments_data
        self.segments_labels = segments_labels
        
    def balance_classes(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Undersample to balance classes"""
        print("Before balancing:", dict(zip(*np.unique(labels, return_counts=True))))   # Before balancing: {0: 69462, 1: 3600, 2: 120} 2 seizure, each 120s = win_size(2s) * #win(60)
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = np.min(counts)
        print("data.shape in balance_classes (expecting: [#seg, 4, 512])", data.shape)      # [#seg, 4, 512]
        print("labels.shape in balance_classes (expecting: [#seg])", labels.shape)  # [#seg]

        balanced_data = []
        balanced_labels = []

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            
            selected_indices = np.random.choice(indices, min_count, replace=False)      # min_count could be larger than the real min count of label: oversample for the minority class
            balanced_data.append(data[selected_indices, :, :])
            balanced_labels.append(labels[selected_indices])
        
        data_balanced = np.concatenate(balanced_data, axis=0)
        labels_balanced = np.concatenate(balanced_labels)
        print("After balancing:", dict(zip(*np.unique(labels_balanced, return_counts=True))))
        print(f"Balanced data shape: {data_balanced.shape}")  # 检查平衡后数据的维度

        return data_balanced, labels_balanced

    def segment_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Segment data into windows with specified size and overlap."""
        num_channels, num_samples = self.data.shape
        step = self.window_size  # No overlap as per requirement
        
        # Calculate number of complete windows
        num_windows = (num_samples - self.window_size) // step + 1
        
        # Initialize arrays for segments
        segments_data = np.zeros((num_windows, num_channels, self.window_size))
        segments_labels = np.zeros(num_windows, dtype=int)
        
        # Extract segments
        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + self.window_size
            
            # Extract data segment
            segments_data[i] = self.data[:, start_idx:end_idx]
            
            # Determine label for segment (majority vote)
            segment_labels = self.labels[start_idx:end_idx]
            unique_labels, counts = np.unique(segment_labels, return_counts=True)
            segments_labels[i] = unique_labels[np.argmax(counts)]   # TODO: majority voting vs over 50% majority
        
        return segments_data, segments_labels
    
    def __len__(self):
        print("len(self.segments_data)", len(self.segments_data))
        print("len(self.segments_labels)", len(self.segments_labels))
        assert len(self.segments_data) == len(self.segments_labels), "Mismatch in data and label lengths"
        return len(self.segments_data)
        # return len(self.segments_labels)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.segments_data[idx]).float(), torch.tensor(self.segments_labels[idx]).long()

def preprocess_data(data: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply all preprocessing steps in place."""
    
    # Normalize each channel
    # replace NaN and extreme values as mean
    nan_index = []
    for ch in range(data.shape[0]):
        only_nan_indices = np.where(np.isnan(data[ch]))[0]
        
        data[ch][data[ch] > 9000] = np.nan  # TODO: remove extreme values
        data[ch][data[ch] < -9000] = np.nan
        nan_indices = np.where(np.isnan(data[ch]))[0]
        nan_index.append(nan_indices)

    all_nan_indices = nan_index[0]
    for i in range(1, len(nan_index)):
        all_nan_indices = np.union1d(all_nan_indices, nan_index[i])

    # First set all the all_nan_indices positions to NaN to ensure that the NaN positions of all channels are consistent.
    mean_1 = np.nanmean(data[0])
    mean_2 = np.nanmean(data[1])
    mean_3 = np.nanmean(data[2])
    mean_4 = np.nanmean(data[3])
    # print("mean1234", mean_1, mean_2, mean_3, mean_4)
    
    data[:, all_nan_indices] = np.nan   # remove all nan and extreme values
    for ch in range(data.shape[0]):
        mean = np.nanmean(data[ch])
        # print("np.nanmean: (based on the nan-removed length)", mean)
        data[ch] = np.nan_to_num(data[ch], nan=mean)       # for each channel: Replace NaN and extreme values with np.nanmean
        std = np.std(data[ch])

        assert std != 0, "std can't be 0"
        if std != 0:
            data[ch] = (data[ch] - mean) / std          # Z-score transformation
        
    # Apply filters for each channel first: avoid high-frequency components that could cause aliasing when downsampling.
    # Compute expected length after resampling
    resampled_data = []
    for ch in range(data.shape[0]):
        # Bandpass filter (0.5-120 Hz)
        data[ch] = apply_bandpass_filter(data[ch], 0.5, 120, fs)
        
        # Notch filters for power line interference and its harmonics
        for freq in [50, 60]:  # Both 50Hz and 60Hz to be safe
            data[ch] = apply_notch_filter(data[ch], freq, Q=30, fs=fs)

        # Resample thereafter
        resampled_data.append(resample_data(data[ch], fs, TARGET_FS))

    resampled_data = np.array(resampled_data, dtype=object)

    return resampled_data, nan_index

def balance_classes(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Balance classes by undersampling majority classes."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)
    
    balanced_data = []
    balanced_labels = []
    
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        selected_indices = np.random.choice(indices, min_count, replace=False)
        balanced_data.append(data[selected_indices])
        balanced_labels.append(labels[selected_indices])
    
    return np.concatenate(balanced_data), np.concatenate(balanced_labels)

# remove intervals containing nan index, require intervals to be at least 2min
def remove_nan_intervals(intervals: P.Interval, nan_intervals: P.Interval) -> P.Interval:
    clean_intervals = P.empty()
    min_length = 120 * TARGET_FS  # mininal length
    for interval in intervals:
        remaining = interval - nan_intervals
        for sub_interval in remaining:
            if sub_interval.upper - sub_interval.lower + 1 >= min_length:
                clean_intervals |= sub_interval
    return clean_intervals


def get_seizure_labels(seizure_info: np.ndarray, fs: int, total_samples: int, channel_data: np.ndarray, nan_index: np.ndarray) -> np.ndarray:
    """Generate labels for all time points based on seizure information."""
    labels = np.full(total_samples, -1, dtype=int)  # Initialize with -1 to identify unlabeled regions
    
    # Convert seizure onset times to sample indices
    onset_indices = [int(onset[0][0] * fs) for onset in seizure_info]
    
    # Create intervals for each class
    ictal_intervals = P.empty()
    preictal_intervals = P.empty()
    interictal_intervals = P.empty()
    
    for i, onset_idx in enumerate(onset_indices):
        # Ictal period (2 minutes)
        ictal_start = onset_idx
        ictal_end = min(onset_idx + int(2 * 60 * fs), total_samples)
        ictal_intervals |= P.closed(ictal_start, ictal_end)
        
        # Pre-ictal period (1 hour before onset)
        preictal_start = max(0, onset_idx - int(1 * 60 * 60 * fs))
        preictal_end = onset_idx
        preictal_intervals |= P.closed(preictal_start, preictal_end)
        # TODO: for HUP273c, the starting time of two onsets are [233295.095421000], [245129.227654]
        #       interval between the two onsets was only 3.287h
        #       hard to find inter-ictal

        # find if there's proper inter-ictal before the first onset
        if i == 0 and ictal_start - int(3 * 60 * 60 * fs) > 0:
            interictal_start = 0
            interictal_end = ictal_start - int(3 * 60 * 60 * fs)
            if interictal_end > interictal_start:
                interictal_intervals |= P.closed(interictal_start, interictal_end)


        # Inter-ictal period (3h after onset and 4h before next onset)
        if i < len(onset_indices) - 1:
            next_onset = onset_indices[i + 1]
            interictal_start = min(total_samples, ictal_end + int(3 * 60 * 60 * fs))
            interictal_end = max(interictal_start, min(total_samples, next_onset - int(4 * 60 * 60 * fs)))
            if interictal_end > interictal_start:
                interictal_intervals |= P.closed(interictal_start, interictal_end)
    
    # Remove empty intervals and merge overlapping ones
    ictal_intervals = merge_overlapping_intervals(remove_empty_intervals(ictal_intervals))
    preictal_intervals = merge_overlapping_intervals(remove_empty_intervals(preictal_intervals))
    interictal_intervals = merge_overlapping_intervals(remove_empty_intervals(interictal_intervals))
  
    # Apply labels
    # get NaN intervals
    nan_intervals = P.empty()

    if len(nan_index) > 0:
        start, end = nan_index[0], nan_index[0]
        for i in range(1, len(nan_index)):
            if nan_index[i] == end + 1:   # continuous NaN
                end = nan_index[i]
            else:
                nan_intervals |= P.closed(start, end)
                start, end = nan_index[i], nan_index[i]
        nan_intervals |= P.closed(start, end)       # add the last intervals

    clean_interictal_intervals = remove_nan_intervals(interictal_intervals, nan_intervals)
    clean_preictal_intervals = remove_nan_intervals(preictal_intervals, nan_intervals)
    clean_ictal_intervals = remove_nan_intervals(ictal_intervals, nan_intervals)

    # print(f"Clean Interictal Intervals: {clean_interictal_intervals}")
    # print(f"Clean Preictal Intervals: {clean_preictal_intervals}")
    # print(f"Clean Ictal Intervals: {clean_ictal_intervals}")

    # label
    for interval in clean_interictal_intervals:
        # print("sample slice:", channel_data[int(interval.lower):int(interval.upper)+1][:5])
        segment = channel_data[int(interval.lower): int(interval.upper) + 1]
        segment = np.asarray(segment, dtype=np.float64)  # 强制转换
        assert not np.isnan(segment).any(), \
            f"Interictal: Interval [{interval.lower}, {interval.upper}] contains NaN in seizure segment"
        labels[int(interval.lower): int(interval.upper) + 1] = 0

    for interval in clean_preictal_intervals:
        segment = channel_data[int(interval.lower): int(interval.upper) + 1]
        segment = np.asarray(segment, dtype=np.float64)  # 强制转换
        assert not np.isnan(segment).any(), \
            f"preictal: Interval [{interval.lower}, {interval.upper}] contains NaN in seizure segment"
        labels[int(interval.lower): int(interval.upper) + 1] = 1

    for interval in clean_ictal_intervals:
        segment = channel_data[int(interval.lower): int(interval.upper) + 1]
        segment = np.asarray(segment, dtype=np.float64)  # 强制转换
        assert not np.isnan(segment).any(), \
            f"Ictal: Interval [{interval.lower}, {interval.upper}] contains NaN in seizure segment"
        labels[int(interval.lower): int(interval.upper) + 1] = 2

    return labels

def resample_data(data: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Resample data to target sampling frequency."""
    if orig_fs != target_fs:
        return resampy.resample(data, orig_fs, target_fs)
    return data


def load_patient_data(data_dir: str, ann_fname: str) -> Tuple[Dataset, Dataset]:
    """Load and process data for a single patient."""
    # Load annotation file
    anno = sio.loadmat(ann_fname)
    fs = anno['fs'].item()
    seizure_info = anno['tszr']
    
    # Load all channels
    all_data = []
    
    for channel_name, mat_file in CHANNEL_DICT.items():
        data_path = data_dir + mat_file
        
        try:
            raw_data = mat73.loadmat(data_path)
        except TypeError:
            raw_data = sio.loadmat(data_path)

        channel_data = raw_data['eeg']
        
        # channel_data.shape checked, it's been resampled TARGET_FS (256Hz)
        all_data.append(channel_data)
    
    # Stack all channels
    data = np.stack(all_data, axis=0)  # Shape: (num_channels, num_samples)
    data, all_chan_nan_index = preprocess_data(data, fs)
    # Generate labels
    labels = get_seizure_labels(seizure_info, TARGET_FS, data.shape[1], data[0], all_chan_nan_index[0])

    # Remove segments with undefined labels (-1)
    valid_indices = labels != -1
    data = data[:, valid_indices]
    labels = labels[valid_indices]

    # Create datasets with windowing
    train_size = int(0.8 * len(labels))
    
    # Split maintaining temporal continuity
    train_data = data[:, :train_size]
    train_labels = labels[:train_size]
    test_data = data[:, train_size:]
    test_labels = labels[train_size:]

    # Create datasets
    window_size_samples = int(WINDOW_SIZE_SEC * TARGET_FS)
    train_dataset = SeizureDataset(train_data, train_labels, window_size_samples, TARGET_FS, balance=True)
    test_dataset = SeizureDataset(test_data, test_labels, window_size_samples, TARGET_FS, balance=False)
    print("test_dataset", dict(zip(*np.unique(test_labels, return_counts=True))))

    return train_dataset, test_dataset

def get_dataloaders(data_dir: str, ann_fname: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders for a patient."""
    train_dataset, test_dataset = load_patient_data(data_dir, ann_fname)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

def patient_person_extract(data_dir: str) -> Tuple[DataLoader, DataLoader]:
    """
    Process data from all patients and create combined train and test dataloaders.
    
    Args:
        data_dir: Base directory containing all patient data
    Returns:
        combined_train_dataset, combined_test_dataset for all patients combined
    """
    subjects_with_anno = ["262b", "267", "269", "270", "271", "272", "273c"]
    # subjects_with_anno = ["272"]
    data_fnames = [f for f in glob.glob(data_dir + "HUP*_phaseII/") if any(sub in f for sub in subjects_with_anno)]
    ann_fnames = [f for f in glob.glob(data_dir + "HUP*_phaseII/HUP*_phaseII.mat") if any(sub in f for sub in subjects_with_anno)]
    data_fnames.sort()
    ann_fnames.sort()

    all_train_datasets = []
    all_test_datasets = []
    
    # Process each patient's data
    for data_fname, ann_fname in zip(data_fnames, ann_fnames):
        print(f"For patient: {ann_fname}")
        train_dataset, test_dataset = load_patient_data(data_fname, ann_fname)
        all_train_datasets.append(train_dataset)
        all_test_datasets.append(test_dataset)

        
    # Combine datasets
    combined_train_dataset = ConcatDataset(all_train_datasets)
    combined_test_dataset = ConcatDataset(all_test_datasets)
    
    return combined_train_dataset, combined_test_dataset


if __name__ == "__main__":
    
    # Example usage for multiple patients
    data_dir = "/home/yqxu/projects/def-xilinliu/data/UPenn_data/"
    # data_dir = "/Users/xuyaqian/Documents/earEEG-epilepsy/data/"
    train_set, test_set = patient_person_extract(data_dir)
    
    torch.save(train_set, "train_set_all.pth", pickle_protocol=4)
    torch.save(test_set, "test_set_all.pth", pickle_protocol=4)
    print(f"train_set.pth & test_set.pth ALL saved")
    
    # Create dataloaders
    batch_size = 512        # Batch size for dataloaders
    num_workers = 2       # Number of workers for dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,          # TODO
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print dataset information
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Example of accessing data
    for batch_data, batch_labels in iter(train_loader):
        print(f"Batch shape: {batch_data.shape}")       # Should be (batch_size, 4, sequence_length)     # sequence_length (2s) * 256
        print(f"Labels shape: {batch_labels.shape}")    # Should be (batch_size,)
        unique_labels, counts = np.unique(batch_labels.numpy(), return_counts=True)
        print("Class distribution in batch:", dict(zip(unique_labels, counts)))
    for batch_data, batch_labels in iter(test_loader):
        print(f"Test Batch shape: {batch_data.shape}")       # Should be (batch_size, 4, sequence_length)     # sequence_length (2s) * 256
        print(f"Labels shape: {batch_labels.shape}")    # Should be (batch_size,)
        unique_labels, counts = np.unique(batch_labels.numpy(), return_counts=True)
        print("Class distribution in batch:", dict(zip(unique_labels, counts)))

