import parselmouth
from .duration_energy import extract_duration_energy
from .pitch_features import extract_pitch_features
from .jitter_shimmer import extract_jitter_shimmer
from .formant_features import extract_formant_features
from .harmonicity_features import extract_harmonicity_features
from .intensity_features import extract_intensity_features
from .mfcc_features import extract_mfcc_features
from .spectrum_features import extract_spectrum_features

def extract_all_features(audio_path):
    snd = parselmouth.Sound(audio_path)

    features = {}
    features.update(extract_duration_energy(snd))
    features.update(extract_pitch_features(snd))
    features.update(extract_jitter_shimmer(snd))
    features.update(extract_formant_features(snd))
    features.update(extract_harmonicity_features(snd))
    features.update(extract_intensity_features(snd))
    features.update(extract_mfcc_features(snd))
    features.update(extract_spectrum_features(snd))

    print(audio_path, ' 완료')
    return features
