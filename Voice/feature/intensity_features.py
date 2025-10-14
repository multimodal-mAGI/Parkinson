import parselmouth

def extract_intensity_features(snd):
    intensity = snd.to_intensity()
    mean_intensity = intensity.get_average(0, 0, parselmouth.Intensity.AveragingMethod.ENERGY)
    min_intensity = parselmouth.praat.call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = parselmouth.praat.call(intensity, "Get maximum", 0, 0, "Parabolic")
    return {
        'mean_intensity': mean_intensity,
        'min_intensity': min_intensity,
        'max_intensity': max_intensity
    }
