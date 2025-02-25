# Hyperparams

## STFT configuration

```py
def _make_stft_cfg(hop_length, win_length=None):
    if win_length is None:
        win_length = 4 * hop_length
    
    n_fft = 2 ** (win_length - 1).bit_length()
    
    return dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
```

- Computes STFT parameters dynamically.
- `win_length` defaults to 4 times `hop_length` if not specified.
- `n_fft` (FFT size) is set to the next power of 2 greater than or equal to win_length.
    - Ensures efficient FFT computation.

---

## HPram class

The `HParams` class stores all hyperparameters as a dataclass.

### Dataset path

```py
@dataclass(frozen=True)
class HParams:
    fg_dir: Path = Path("data/fg")
    bg_dir: Path = Path("data/bg")
    rir_dir: Path = Path("data/rir")
    load_fg_only: bool = False
    praat_augment_prob: float = 0
```

- Defines paths for foreground (`fg`), background (`bg`), and room impulse response (`rir`) datasets.
- `load_fg_only`: If True, only the foreground dataset is used.
- `praat_augment_prob`: Probability of using Praat-based audio augmentation.

### Audio processing settings

```py
wav_rate: int = 44_100  # Sample rate (Hz)
n_fft: int = 2048  # FFT window size
win_size: int = 2048  # Window size for STFT
hop_size: int = 420  # Step size between STFT frames
num_mels: int = 128  # Number of mel filter banks
stft_magnitude_min: float = 1e-4  # Minimum magnitude for STFT
preemphasis: float = 0.97  # Pre-emphasis filter
mix_alpha_range: tuple[float, float] = (0.2, 0.8)  # Range for mixing audio
```

- STFT Parameters (n_fft, win_size, hop_size) define the frequency resolution.
- `num_mels`: Number of mel frequency bins (used in Mel Spectrograms).
- `preemphasis`: Applies a high-pass filter to boost high frequencies.
- `mix_alpha_range`: Controls how strongly different sounds are mixed.

**Example**: If mix_alpha_range = (0.2, 0.8), an audio sample could be weighted 20% to 80% when mixed with another.

### Training settings

```py
nj: int = 64  # Number of parallel jobs (for multiprocessing)
training_seconds: float = 1.0  # Duration of training samples
batch_size_per_gpu: int = 16  # Mini-batch size per GPU
min_lr: float = 1e-5  # Minimum learning rate
max_lr: float = 1e-4  # Maximum learning rate
warmup_steps: int = 1000  # Steps for learning rate warmup
max_steps: int = 1_000_000  # Maximum training steps
gradient_clipping: float = 1.0  # Prevents exploding gradients
```

- Adaptive Learning Rate (LR)
    - Starts from `min_lr` → peaks at `max_lr`.
    - Controlled via `warmup_steps`.
- Gradient Clipping: Prevents extreme gradient values from causing instability.

### DeepSpeed configuration

```py
@property
def deepspeed_config(self):
    return {
        "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": float(self.min_lr)},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": float(self.min_lr),
                "warmup_max_lr": float(self.max_lr),
                "warmup_num_steps": self.warmup_steps,
                "total_num_steps": self.max_steps,
                "warmup_type": "linear",
            },
        },
        "gradient_clipping": self.gradient_clipping,
    }
```

- DeepSpeed is a framework for efficient large-scale model training.
- Uses:
    - Adam optimizer
    - Linear warmup learning rate scheduler (gradually increases LR at the beginning)
    - Gradient clipping for stability.

### STFT configuration getter

```py
@property
def stft_cfgs(self):
    assert self.wav_rate == 44_100, f"wav_rate must be 44_100, got {self.wav_rate}"
    return [_make_stft_cfg(h) for h in (100, 256, 512)]
```

- Ensures that `wav_rate` is 44.1 kHz.
- Generates 3 different STFT configurations using `_make_stft_cfg()`.

Sample output

```py
[
    {'n_fft': 256, 'hop_length': 100, 'win_length': 400},
    {'n_fft': 512, 'hop_length': 256, 'win_length': 1024},
    {'n_fft': 1024, 'hop_length': 512, 'win_length': 2048}
]
```
