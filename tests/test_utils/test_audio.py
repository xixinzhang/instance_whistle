from pathlib import Path

from whistle_prompter import utils
from whistle_prompter.datasets.prepare_spec_img import *


def test_spec_extraction(stem:str):
    audio_file = Path(f"data/audio/{stem}.wav")
    audio = utils.load_audio(audio_file)
    print(audio.shape, audio.dtype, audio.min(), audio.max())
    spec = utils.spectrogram(audio)
    print(spec.shape, spec.dtype, spec.min(), spec.max())

def test_spec_cutting(stem:str):
    audio_file = Path(f"data/audio/{stem}.wav")
    audio = utils.load_audio(audio_file)
    spec = utils.spectrogram(audio)
    segments = utils.cut_sepc(spec)
    print(segments.keys())

def test_segments_dict(stem:str):
    segments_dict = audios_to_segments_dict(f"data/audio/{stem}.wav")
    print(segments_dict.keys())
    print(segments_dict[stem].keys())
    print(segments_dict[stem][0].shape)

def test_spec_img(stem:str, save_dir = f"tests/data/spec_img/", cmap = 'magma'):
    audio_file = Path(f"data/audio/{stem}.wav")
    segments_dict = audios_to_segments_dict(audio_file)
    save_specs_img(segments_dict, save_dir, cmap)



if __name__ == "__main__":
    with open("data/meta.json") as f:
        import json
        meta = json.load(f)

    stem = meta['data']['train'][0]

    print('Testing extraction')
    test_spec_extraction(stem)
    print('Testing cutting')
    test_spec_cutting(stem)
    print('Testing segments dict')
    test_segments_dict(stem)
    print('Testing spec img')
    test_spec_img(stem)

    # test all files
    for split, stems in meta["data"].items():
        for stem in stems:
            print(split, stem)
            test_spec_extraction(stem)
            