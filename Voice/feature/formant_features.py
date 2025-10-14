import numpy as np
import parselmouth

def extract_formant_features(snd):
    formant = snd.to_formant_burg()
    mean_f1 = parselmouth.praat.call(formant, "Get mean", 1, 0, 0, "Hertz")
    mean_f2 = parselmouth.praat.call(formant, "Get mean", 2, 0, 0, "Hertz")
    mean_f3 = parselmouth.praat.call(formant, "Get mean", 3, 0, 0, "Hertz")

    times = np.arange(formant.xmin, formant.xmax, formant.dx)
    bw_f1 = np.mean([formant.get_bandwidth_at_time(1, t) for t in times if not np.isnan(formant.get_bandwidth_at_time(1, t))])
    bw_f2 = np.mean([formant.get_bandwidth_at_time(2, t) for t in times if not np.isnan(formant.get_bandwidth_at_time(2, t))])
    bw_f3 = np.mean([formant.get_bandwidth_at_time(3, t) for t in times if not np.isnan(formant.get_bandwidth_at_time(3, t))])

    return {
        'mean_f1': mean_f1, 'mean_f2': mean_f2, 'mean_f3': mean_f3,
        'mean_bw_f1': bw_f1, 'mean_bw_f2': bw_f2, 'mean_bw_f3': bw_f3
    }
