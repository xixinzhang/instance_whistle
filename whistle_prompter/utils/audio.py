from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torch.nn.functional import pad


def exact_div(x, y):
    assert x % y == 0
    return x // y

# hard-coded audio hyperparameters
WINDOW_MS = 8
HOP_MS = 2
SAMPLE_RATE = 192_000
N_FFT = exact_div(SAMPLE_RATE * WINDOW_MS,  1000)  
HOP_LENGTH = exact_div(SAMPLE_RATE * HOP_MS, 1000)

CHUNK_LENGTH = 3 # second cover whistle length
N_SAMPLES = int(SAMPLE_RATE * CHUNK_LENGTH)
N_FRAMES = exact_div(N_SAMPLES,  HOP_LENGTH) # 1500

FRAME_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)
NUM_FREQ_BINS = exact_div(N_FFT, 2) + 1
FREQ_BIN_RESOLUTION = SAMPLE_RATE / N_FFT  # 125 Hz

TOP_DB = 80
AMIN = 1e-10


def load_audio(file:Union[str, Path]) -> torch.Tensor:
    """Load audio wave file to tensor
    
    Args:
        file: audio file path
    Return:
        waveform: (1, L)
    """
    try:
        waveform, sample_rate = torchaudio.load(file)
    except RuntimeError as e:
        import librosa
        waveform, sample_rate = librosa.load(file, sr=None)
        waveform = torch.tensor(waveform).unsqueeze(0)
    waveform =waveform/ torch.max(torch.abs(waveform)) # normalize to [-1, 1]
    return waveform  # (1, L)


def spectrogram(waveform: torch.Tensor, device: Optional[Union[str, torch.device]] = None):
    """Compute spectrogram from waveform
    
    Args:
        waveform: (1, L)
        device: device to run the computation

    Return:
        spectrogram: (F, T) in range [-TOP_DB, 0]
    """

    window = torch.hann_window(N_FFT).to(device=device)
    spec = F.spectrogram(
        waveform,
        pad=0,
        window=window,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        power=2,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )
    # spec = F.amplitude_to_DB(spec, multiplier=10, amin=AMIN, db_multiplier = 1, top_db = TOP_DB)
    # bound spect to [-TOP_DB, 0]
    spec = F.amplitude_to_DB(spec, multiplier=10, amin=AMIN, db_multiplier = torch.max(spec).log10(), top_db = TOP_DB)
    spec = spec[..., :-1].squeeze(0) # drop last frame
    spec = torch.flipud(spec) # flip frequency axis, low freq at the bottom
    return spec # (F, T)  [-TOP_DB, 0]

def cut_sepc(spec: torch.Tensor, overlap: float)-> dict[int, np.ndarray]:
    """Segment spectrogram into chunks/segments with/without overlap
    
    Args:
        spec: (1, F, T)
    Return:
        segments:{start_frame: segment}
            segment: (1, F, N_FRAMES)
            start_frame: start frame of the segment
    """
    segments = dict()
    overlap_frames = int(overlap * N_FRAMES)
    for start_frame in range(0, spec.shape[1], N_FRAMES - overlap_frames):
        segment = spec[:, start_frame : start_frame + N_FRAMES]
        if segment.shape[-1] < N_FRAMES:
            segment = pad(segment, (0, N_FRAMES - segment.shape[-1]))

        segments[start_frame] = segment.numpy()
    return segments

def normalize_spec_img(spec: torch.Tensor) -> torch.Tensor:
    """Normalize spectrogram to [0, 1] to be compatible with image format"""
    spec = (spec + TOP_DB) / TOP_DB
    return spec

def apply_colormap(spec:np.array, cmap: str = 'bone') -> np.array:
    """Apply colormap to spectrogram
    
    Args:
        spec: (F, N_FRAMES) one channel spectrogram
        cmap: colormap name
    Return:
        colored_spec: (F, N_FRAMES, 3) RGB colored spectrogram 
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap)
    colored_spec = cmap(spec)
    colored_spec = colored_spec[..., :3] # drop alpha channel
    return colored_spec
