import numpy as np
import parselmouth

def extract_spectrum_features(snd):
    duration = snd.duration
    spectrum = snd.to_spectrum()
    mean_power = parselmouth.praat.call(spectrum, "Get band energy", 0, 0) / duration

    spectrogram = snd.to_spectrogram()
    spec_times = np.arange(spectrogram.xmin, spectrogram.xmax, spectrogram.dx)
    spec_freqs = np.arange(spectrogram.ymin, spectrogram.ymax, spectrogram.dy)
    spec_powers = [spectrogram.get_power_at(t, f) for t in spec_times for f in spec_freqs]
    mean_spec_power = np.mean([p for p in spec_powers if not np.isnan(p)])

    return {'mean_power': mean_power, 'mean_spec_power': mean_spec_power}
