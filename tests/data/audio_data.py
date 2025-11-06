
import os
import wave
import contextlib
import argparse
from whistle_prompter.utils.annotation import load_tonal_reader


def get_wav_stats_and_stems(root_folder, exclude_stems=None):
    total_length = 0.0
    wav_count = 0
    exclude_stems = set(exclude_stems or [])
    included_stems = []

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                stem = os.path.splitext(fname)[0]
                if stem in exclude_stems:
                    continue
                wav_count += 1
                included_stems.append(stem)
                filepath = os.path.join(dirpath, fname)
                try:
                    with contextlib.closing(wave.open(filepath, 'r')) as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                        total_length += duration
                except wave.Error as e:
                    print(f"Could not read {filepath}: {e}")
    return wav_count, total_length, sorted(included_stems)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get stats for .wav files in a folder.")
    parser.add_argument(
        "folder",
        type=str,
        default=os.path.expanduser("~") + "/Desktop/projects/whistle_prompter/data/dclde/audio",
        help="Target folder to search for .wav files."
    )
    parser.add_argument(
        "--count-annotations",
        action='store_true',
        help="Count the number of annotations for each .wav file by loading the corresponding .bin file. By default, the .bin file is searched in the same folder as the .wav file."
    )
    parser.add_argument(
        "--bin-folder",
        type=str,
        default=None,
        help="Optional folder to search for .bin files. If not provided, will use the same folder as the .wav file."
    )
    parser.add_argument(
        "--exclude",
        nargs='*',
        default=[],
        help="List of .wav file stems to exclude (filenames without extension)."
    )
    parser.add_argument(
        "--list-stems",
        action='store_true',
        help="List all included .wav file stems, one per line after summary."
    )
    args = parser.parse_args()

    count, length, stems = get_wav_stats_and_stems(args.folder, args.exclude)
    print(f"Found {count} .wav files")
    print(f"Total length: {length} seconds ({length/60:.2f} minutes)")

    if args.count_annotations:
        print("\nTotal annotation count for included .wav stems:")
        bin_folder = args.bin_folder if args.bin_folder else args.folder
        total_annos = 0
        for stem in stems:
            bin_path = None
            # Search for .bin file with same stem in bin_folder
            for dirpath, _, filenames in os.walk(bin_folder):
                if f"{stem}.bin" in filenames:
                    bin_path = os.path.join(dirpath, f"{stem}.bin")
                    break
            if bin_path and os.path.isfile(bin_path):
                try:
                    annos = load_tonal_reader(bin_path)
                    total_annos += len(annos)
                except Exception as e:
                    import pdb; pdb.set_trace()
                    print(f"Error loading {stem}.bin: {e}")
            else:
                print(f"{stem}.bin not found in {bin_folder}")
        print(f"Total annotations: {total_annos}")
    if args.list_stems:
        print("Included stems:")
        for stem in stems:
            print(stem)
