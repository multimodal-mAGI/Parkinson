import numpy as np
import parselmouth

def extract_mfcc_features(snd):
    mfcc = snd.to_mfcc(number_of_coefficients=12)
    mfcc_means = {}
    for i in range(1, 13):
        values = [mfcc.get_value_in_frame(f, i) for f in range(1, mfcc.get_number_of_frames() + 1)
                  if not np.isnan(mfcc.get_value_in_frame(f, i))]
        mfcc_means[f'mfcc_{i}'] = np.mean(values) if values else np.nan
    return mfcc_means
