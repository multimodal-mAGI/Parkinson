import numpy as np
import librosa
import torch


class AudioPreprocessor:
    """오디오 전처리 클래스"""
    
    def __init__(self, target_length=32000, sample_rate=16000, n_mels=64, n_mfcc=64):
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
    
    def load_and_preprocess_audio(self, audio_paths):
        """실제 오디오 파일들을 로드하고 전처리"""
        processed_data = {
            'cnn': [],
            'rnn': [],
            'transformer': [],
            'hybrid': []
        }
        
        print(f"오디오 파일 {len(audio_paths)}개 처리 중...")
        
        for i, audio_path in enumerate(audio_paths):
            if i % 50 == 0:
                print(f"진행률: {i}/{len(audio_paths)}")
                
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                
                # 길이 정규화
                if len(audio) > self.target_length:
                    audio = audio[:self.target_length]
                else:
                    audio = np.pad(audio, (0, self.target_length - len(audio)), 'constant')
                
                audio = np.nan_to_num(audio, nan=0.0)
                
                # CNN용 멜 스펙트로그램
                mel_spec = self._extract_mel_spectrogram(audio, sr)
                processed_data['cnn'].append(mel_spec)
                
                # RNN용 MFCC
                mfcc = self._extract_mfcc(audio, sr)
                processed_data['rnn'].append(mfcc)
                
                # Transformer용 원본 오디오
                processed_data['transformer'].append(audio)
                
                # Hybrid용 MFCC (형태 변환)
                mfcc_for_hybrid = self._prepare_mfcc_for_hybrid(audio, sr)
                processed_data['hybrid'].append(mfcc_for_hybrid)
                
            except Exception as e:
                print(f"오디오 파일 {audio_path} 처리 중 오류: {e}")
                # 기본 제로 데이터로 채우기
                processed_data['cnn'].append(np.zeros((self.n_mels, self.n_mels)))
                processed_data['rnn'].append(np.zeros((self.n_mels, self.n_mels)))
                processed_data['transformer'].append(np.zeros(self.target_length))
                processed_data['hybrid'].append(np.zeros((self.n_mels, self.n_mels)))
        
        # numpy 배열로 변환
        for key in processed_data:
            processed_data[key] = np.array(processed_data[key])
        
        print("오디오 전처리 완료!")
        return processed_data
    
    def _extract_mel_spectrogram(self, audio, sr):
        """멜 스펙트로그램 추출"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec.shape[1] != self.n_mels:
            mel_spec = np.resize(mel_spec, (self.n_mels, self.n_mels))
        mel_spec = np.nan_to_num(mel_spec, nan=0.0)
        return mel_spec
    
    def _extract_mfcc(self, audio, sr):
        """MFCC 추출"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = mfcc.T
        if mfcc.shape[0] != self.n_mels:
            mfcc = np.resize(mfcc, (self.n_mels, self.n_mels))
        mfcc = np.nan_to_num(mfcc, nan=0.0)
        return mfcc
    
    def _prepare_mfcc_for_hybrid(self, audio, sr):
        """Hybrid 모델용 MFCC 준비"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = mfcc.T
        if mfcc.shape[0] != self.n_mels:
            mfcc_for_hybrid = np.resize(mfcc.T, (self.n_mels, self.n_mels))
        else:
            mfcc_for_hybrid = mfcc.T
        mfcc_for_hybrid = np.nan_to_num(mfcc_for_hybrid, nan=0.0)
        return mfcc_for_hybrid


def prepare_batch_data(processed_data, model_name, indices, device):
    """배치 데이터 준비"""
    if model_name == 'cnn':
        batch_data = torch.FloatTensor(processed_data['cnn'][indices]).unsqueeze(1).to(device)
    elif model_name == 'rnn':
        batch_data = torch.FloatTensor(processed_data['rnn'][indices]).to(device)
    elif model_name == 'transformer':
        batch_data = torch.FloatTensor(processed_data['transformer'][indices]).to(device)
    else:  # hybrid
        batch_data = torch.FloatTensor(processed_data['hybrid'][indices]).to(device)
    
    return batch_data