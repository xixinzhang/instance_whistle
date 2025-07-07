import os
import glob
import torch
import yaml
from rich import print as rprint

############## meta data of split of dataset ##############
# meta = {}
# root_dir = os.path.expanduser("~/storage/DCLDE/whale_whistle")
# classes = ['bottlenose', 'common', 'melon-headed', 'spinner']
# filenames = []
# for s in classes[:2]:
#     filenames.extend(glob.glob(os.path.join(root_dir, s, "*.wav")))
# stems = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
# meta['test'] = stems

# filenames = []
# for s in classes[2:]:
#     filenames.extend(glob.glob(os.path.join(root_dir, s, "*.wav")))
# stems = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
# meta['train'] = stems

# with open('meta.yaml', 'w') as f:
#     yaml.safe_dump(meta, f)


meta1 = yaml.safe_load(open('/home/xzhang3906/Desktop/projects/whistle_prompter/data/cross/meta.yaml'))
# meta2 = yaml.safe_load(open('/home/xzhang3906/Desktop/projects/whistle_prompter/meta.yaml'))
# for k in meta1['test']:
#     if k not in meta2['test']:
#         print(k)

############## Check the audio duration and annotation length ##############
# from whistle_prompter import utils
# for stem in meta1['test']:
#     bin_path = os.path.join('data/cross/anno', f"{stem}.bin")
#     annots = utils.load_annotation(bin_path)
#     min_start = float('inf')
#     max_end = float('-inf')
#     for ann in annots:
#         start, end = min(ann[:, 0]), max(ann[:, 0])
#         min_start = min(min_start, start)
#         max_end = max(max_end, end)


#     audio_path = os.path.join('data/cross/audio', f"{stem}.wav")
#     import librosa
#     waveform, sample_rate = librosa.load(audio_path, sr=None)
#     duration = len(waveform) / sample_rate
#     if min_start > 30 or max_end < duration - 30 or max_end - min_start < 30:
#         print(f"{stem}: {min_start:.2f} - {max_end:.2f}, duration: {duration:.2f}")

# for stem in meta1['train']:
#     bin_path = os.path.join('data/cross/anno', f"{stem}.bin")
#     annots = utils.load_annotation(bin_path)
#     min_start = float('inf')
#     max_end = float('-inf')
#     for ann in annots:
#         start, end = min(ann[:, 0]), max(ann[:, 0])
#         min_start = min(min_start, start)
#         max_end = max(max_end, end)

#     audio_path = os.path.join('data/cross/audio', f"{stem}.wav")
#     import librosa
#     waveform, sample_rate = librosa.load(audio_path, sr=None)
#     duration = len(waveform) / sample_rate
#     if min_start > 30 or max_end < duration - 30 or max_end - min_start < 30:
#         from rich import print
#         print(f"{stem}: {min_start:.2f} - {max_end:.2f}, duration: {duration:.2f}")




############# Check Db stats ##############
from whistle_prompter.utils.audio import *
# for stem in meta1['train']+meta1['test']:
for stem in ['QX-Dc-CC0604-TAT25-060413-215524']:
    audio_path = os.path.join('data/cross/audio', f"{stem}.wav")
    waveform = load_audio(audio_path)
    window = torch.hann_window(N_FFT)
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
    import pdb; pdb.set_trace()
    # rprint(f"{stem}: {torch.min(spec).item():.2f} - {torch.max(spec).item():.2f}, mean: {torch.mean(spec).item():.2f}, std: {torch.std(spec).item():.2f}")
    spec = F.amplitude_to_DB(spec, multiplier=10, amin=1e-20, db_multiplier = 1)
    spec = spec[..., :-1].squeeze(0) # drop last frame
    rprint(f"{stem}: {torch.min(spec).item():.2f} - {torch.max(spec).item():.2f}, mean: {torch.mean(spec).item():.2f}, std: {torch.std(spec).item():.2f}")
