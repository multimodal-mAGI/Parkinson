import parselmouth

def extract_duration_energy(snd):
    duration = snd.duration
    total_energy = snd.get_energy(0, snd.duration)
    return {'duration': duration, 'total_energy': total_energy}
