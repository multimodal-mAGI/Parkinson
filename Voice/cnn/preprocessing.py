import numpy as np
import librosa
import pywt
from scipy.signal import wiener
from scipy.ndimage import median_filter


class CNNAudioPreprocessor:
    """CNN 전용 오디오 전처리 (노이즈 제거 강화)"""

    def __init__(self, target_length=32000, sample_rate=16000, n_mels=64):
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def load_and_preprocess_audio(self, audio_paths):
        """오디오 로드 및 전처리"""
        processed_data = []

        print(f"오디오 파일 {len(audio_paths)}개 처리 중...")

        for i, audio_path in enumerate(audio_paths):
            if i % 50 == 0:
                print(f"진행률: {i}/{len(audio_paths)}")

            try:
                # 1. 오디오 로드
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)

                # 2. 노이즈 제거 전처리
                audio = self._apply_noise_reduction(audio, sr)

                # 3. 길이 정규화
                audio = self._normalize_length(audio)

                # 4. Mel Spectrogram 추출
                mel_spec = self._extract_mel_spectrogram(audio, sr)

                processed_data.append(mel_spec)

            except Exception as e:
                print(f"오디오 파일 {audio_path} 처리 중 오류: {e}")
                # 기본 제로 데이터
                processed_data.append(np.zeros((self.n_mels, self.n_mels)))

        processed_data = np.array(processed_data)
        print("오디오 전처리 완료!")

        return processed_data

    def _apply_noise_reduction(self, audio, sr):
        """
        노이즈 제거 파이프라인
        1. Wiener 필터 (가우시안 노이즈 제거)
        2. Hamming 윈도우 (스펙트럼 누설 방지)
        3. Wavelet Transform (노이즈 추정)
        """
        # 1. Wiener 필터 적용 (noise_power=0.01)
        audio = self._apply_wiener_filter(audio, noise_power=0.01)

        # 2. Hamming 윈도우 적용
        audio = self._apply_hamming_window(audio)

        # 3. Wavelet Transform 노이즈 제거
        audio = self._apply_wavelet_denoising(audio)

        # NaN 처리
        audio = np.nan_to_num(audio, nan=0.0)

        return audio

    def _apply_wiener_filter(self, audio, noise_power=0.01):
        """
        Wiener 필터: 가우시안 노이즈 제거
        - 노이즈 추정값을 기반으로 최적 필터링
        - SNR 향상
        """
        try:
            # Wiener 필터 적용 (mysize는 필터 윈도우 크기)
            filtered = wiener(audio, mysize=5, noise=noise_power)
            return filtered
        except Exception as e:
            print(f"Wiener 필터 오류: {e}")
            return audio

    def _apply_hamming_window(self, audio):
        """
        Hamming 윈도우: 스펙트럼 누설 방지
        - 신호 양 끝의 불연속성 제거
        - 주파수 분석 정확도 향상
        """
        try:
            window = np.hamming(len(audio))
            windowed = audio * window
            return windowed
        except Exception as e:
            print(f"Hamming 윈도우 오류: {e}")
            return audio

    def _apply_wavelet_denoising(self, audio):
        """
        Wavelet Transform 노이즈 제거
        - 4단계 다해상도 분해
        - MAD 기반 노이즈 추정
        - Bayes Shrink 임계값 계산
        - 소프트 임계값으로 세부 계수 처리
        """
        try:
            # Wavelet 분해 (4단계, db4)
            coeffs = pywt.wavedec(audio, 'db4', level=4)

            # MAD (Median Absolute Deviation) 기반 노이즈 추정
            sigma = self._estimate_noise_mad(coeffs[-1])

            # Bayes Shrink 임계값 계산
            threshold = self._calculate_bayes_shrink_threshold(coeffs, sigma)

            # 소프트 임계값 적용
            coeffs_thresh = [coeffs[0]]  # approximation 계수는 유지
            for detail in coeffs[1:]:
                coeffs_thresh.append(pywt.threshold(detail, threshold, mode='soft'))

            # 재구성
            denoised = pywt.waverec(coeffs_thresh, 'db4')

            # 길이 맞추기
            if len(denoised) > len(audio):
                denoised = denoised[:len(audio)]
            elif len(denoised) < len(audio):
                denoised = np.pad(denoised, (0, len(audio) - len(denoised)), 'constant')

            return denoised

        except Exception as e:
            print(f"Wavelet 노이즈 제거 오류: {e}")
            return audio

    def _estimate_noise_mad(self, detail_coeffs):
        """MAD (Median Absolute Deviation)로 노이즈 추정"""
        mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        sigma = mad / 0.6745  # 가우시안 분포 가정
        return sigma

    def _calculate_bayes_shrink_threshold(self, coeffs, sigma):
        """Bayes Shrink 임계값 계산"""
        try:
            # 세부 계수들의 분산 계산
            detail_coeffs = np.concatenate([c for c in coeffs[1:]])
            var_y = np.var(detail_coeffs)
            var_x = max(var_y - sigma**2, 0)

            if var_x == 0:
                threshold = sigma
            else:
                threshold = sigma**2 / np.sqrt(var_x)

            return threshold
        except Exception as e:
            print(f"Bayes Shrink 임계값 계산 오류: {e}")
            return sigma

    def _normalize_length(self, audio):
        """오디오 길이 정규화"""
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            audio = np.pad(audio, (0, self.target_length - len(audio)), 'constant')
        return audio

    def _extract_mel_spectrogram(self, audio, sr):
        """Mel Spectrogram 추출"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 크기 조정
        if mel_spec.shape[1] != self.n_mels:
            mel_spec = np.resize(mel_spec, (self.n_mels, self.n_mels))

        mel_spec = np.nan_to_num(mel_spec, nan=0.0)

        return mel_spec
