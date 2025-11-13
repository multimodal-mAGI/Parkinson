import os

def load_audio_dataset(root):
    paths, labels = [], []
    for label_name in ["healthy", "parkinson"]:
        label = 0 if label_name == "healthy" else 1
        folder = os.path.join(root, label_name)
        for f in os.listdir(folder):
            if f.lower().endswith((".wav", ".flac")):
                paths.append(os.path.join(folder, f))
                labels.append(label)
    return paths, labels


def load_mfcc_image_dataset(root):
    paths, labels = [], []
    for label_name in ["healthy", "parkinson"]:
        label = 0 if label_name == "healthy" else 1
        folder = os.path.join(root, label_name)
        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, f))
                labels.append(label)
    return paths, labels
