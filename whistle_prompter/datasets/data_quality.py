import os
import glob
import yaml


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

from whistle_prompter import utils
for stem in meta1['test']:
    bin_path = os.path.join('data/cross/anno', f"{stem}.bin")
    annots = utils.load_annotation(bin_path)
    min_start = float('inf')
    max_end = float('-inf')
    for ann in annots:
        start, end = min(ann[:, 0]), max(ann[:, 0])
        min_start = min(min_start, start)
        max_end = max(max_end, end)


    audio_path = os.path.join('data/cross/audio', f"{stem}.wav")
    import librosa
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    duration = len(waveform) / sample_rate
    if min_start > 30 or max_end < duration - 30 or max_end - min_start < 30:
        from rich import print
        print(f"{stem}: {min_start:.2f} - {max_end:.2f}, duration: {duration:.2f}")

for stem in meta1['train']:
    bin_path = os.path.join('data/cross/anno', f"{stem}.bin")
    annots = utils.load_annotation(bin_path)
    min_start = float('inf')
    max_end = float('-inf')
    for ann in annots:
        start, end = min(ann[:, 0]), max(ann[:, 0])
        min_start = min(min_start, start)
        max_end = max(max_end, end)

    audio_path = os.path.join('data/cross/audio', f"{stem}.wav")
    import librosa
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    duration = len(waveform) / sample_rate
    if min_start > 30 or max_end < duration - 30 or max_end - min_start < 30:
        from rich import print
        print(f"{stem}: {min_start:.2f} - {max_end:.2f}, duration: {duration:.2f}")
    
