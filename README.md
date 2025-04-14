# Seizure_Detection_UPenn_earEEG
----2025-04-14
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

----2025-04-14
   
