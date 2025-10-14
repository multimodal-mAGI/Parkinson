import parselmouth

def extract_harmonicity_features(snd):
    harmonicity = snd.to_harmonicity()
    mean_hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    return {'mean_hnr': mean_hnr}
