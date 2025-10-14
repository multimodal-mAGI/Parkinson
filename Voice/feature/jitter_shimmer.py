import numpy as np
import parselmouth

def extract_jitter_shimmer(snd):
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    num_pulses = parselmouth.praat.call(point_process, "Get number of points")
    is_voiced = num_pulses > 0

    if not is_voiced:
        return {
            'num_pulses': num_pulses,
            'mean_period': np.nan,
            'local_jitter': np.nan, 'abs_jitter': np.nan, 'rap_jitter': np.nan,
            'ppq5_jitter': np.nan, 'ddp_jitter': np.nan,
            'local_shimmer': np.nan, 'abs_shimmer': np.nan,
            'apq3_shimmer': np.nan, 'apq5_shimmer': np.nan,
            'apq11_shimmer': np.nan, 'dda_shimmer': np.nan
        }

    mean_period = parselmouth.praat.call(point_process, "Get mean period", 0, 0, 0.0001, 0.02, 1.3)
    local_jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    abs_jitter = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp_jitter = parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    local_shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    abs_shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5_shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return {
        'num_pulses': num_pulses,
        'mean_period': mean_period,
        'local_jitter': local_jitter, 'abs_jitter': abs_jitter,
        'rap_jitter': rap_jitter, 'ppq5_jitter': ppq5_jitter, 'ddp_jitter': ddp_jitter,
        'local_shimmer': local_shimmer, 'abs_shimmer': abs_shimmer,
        'apq3_shimmer': apq3_shimmer, 'apq5_shimmer': apq5_shimmer,
        'apq11_shimmer': apq11_shimmer, 'dda_shimmer': dda_shimmer
    }
