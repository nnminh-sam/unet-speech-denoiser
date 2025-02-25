# Sound Waveform to Mel Spectrogram

## Spectrogram

A spectrogram is a visual representation of sound that shows how frequencies change over time. It is a 2D plot where:

- X-axis (horizontal) = Time
- Y-axis (vertical) = Frequency
- Color/Intensity = Amplitude (Loudness) of each frequency at a given time

**Relationship with Waveform and Sample Rate:**

- A waveform represents raw audio in the time domain (amplitude vs. time).
- A spectrogram represents audio in the frequency domain (frequency vs. time).
- The sample rate (e.g., 16,000 Hz) determines how many samples per second are taken from the waveform.
- Higher sample rates capture more frequency details but require more computation.

The conversion from a waveform to a spectrogram is done using the Short-Time Fourier Transform (STFT).


## MelSpectrogram class

### Class initialization

```py
self.melspec = TorchMelSpectrogram(
    hp.wav_rate,  # Sample rate of the audio
    n_fft=hp.n_fft,  # Number of FFT points (window size for STFT)
    win_length=hp.win_size,  # Window size for STFT
    hop_length=hp.hop_size,  # Step size between STFT windows
    f_min=0,  # Minimum frequency
    f_max=hp.wav_rate // 2,  # Maximum frequency (Nyquist frequency)
    n_mels=hp.num_mels,  # Number of Mel filter banks
    power=1,  # Power of the STFT (1 = magnitude, 2 = power)
    pad_mode="constant",
    norm="slaney",
    mel_scale="slaney",
)
```

- Uses STFT (Short-Time Fourier Transform) to convert waveform → frequency domain.
- Applies a Mel filter bank to get a Mel spectrogram.
- The Mel scale mimics human perception by focusing more on lower frequencies.

**Key params:**

- `n_fft`: The size of the FFT window (higher values = better frequency resolution, worse time resolution).
- `hop_length`: The step size between windows (lower values = more overlap, smoother time resolution).
- `n_mels`: Number of Mel bands (fewer = coarse, more = detailed).

### Processing audio waveform

```py
def forward(self, wav, pad=True):
```

- Takes waveform input (wav) in shape [B, T] (batch size, time steps).
- Checks if the data is on Apple's Metal (MPS) backend (wav.is_mps).
- Applies pre-emphasis filtering (high-pass filter) to boost high frequencies:

    ```py
    wav = wav[..., 1:] - self.preemphasis * wav[..., :-1]
    ```

    This reduces noise and improves spectral balance.

### Compute Mel Spectrogram

```py
mel = self.melspec(wav)
```

- Converts the waveform into a Mel spectrogram.

### Convert to Decibels (Amplitude → Log Scale)

```py
mel = self._amp_to_db(mel)
```

- Audio is usually in linear amplitude values, but we convert to decibels (log scale) because human perception is logarithmic.
- This is done using:

    ```py
    def _amp_to_db(self, x):
    return x.clamp_min(self.hp.stft_magnitude_min).log10() * 20
    ```

    This applies 20 × log10(amplitude).

### Normalize the Spectrogram

```py
mel_normed = self._normalize(mel)
```

- Normalizes the spectrogram values to a fixed range (between 0 and 1) for stable training.
- Uses

    ```py
    def _normalize(self, s, headroom_db=15):
    return (s - self.min_level_db) / (-self.min_level_db + headroom_db)
    ```

    Ensures that all spectrogram values fit into a usable scale.

