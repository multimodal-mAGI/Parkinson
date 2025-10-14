import numpy as np
import parselmouth

def extract_pitch_features(snd):
    pitch = snd.to_pitch()
    freqs = pitch.selected_array['frequency']
    valid_f0 = freqs[freqs > 0]

    mean_f0 = np.mean(valid_f0) if valid_f0.size > 0 else np.nan
    min_f0 = np.min(valid_f0) if valid_f0.size > 0 else np.nan
    max_f0 = np.max(valid_f0) if valid_f0.size > 0 else np.nan

    total_frames = freqs.size
    voiced_percentage = (valid_f0.size / total_frames * 100) if total_frames > 0 else np.nan

    return {
        'mean_f0': mean_f0,
        'min_f0': min_f0,
        'max_f0': max_f0,
        'voiced_percentage': voiced_percentage
    }
