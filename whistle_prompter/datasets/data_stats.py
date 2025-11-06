from collections import defaultdict
import numpy as np
import librosa
import os

from whistle_prompter.utils.annotation import load_tonal_reader


#######################################
# Compute data stats for original data
#######################################
origin_data_dir = "data/whale_whistle"
origin_data_stats = {}
total_file_count = 0
total_duration = 0.0
total_whistle_count = 0
for subdir in os.listdir(origin_data_dir):
    if os.path.isfile(subdir):
        continue
    dirpath = os.path.join(origin_data_dir, subdir)
    dir_name = os.path.basename(dirpath)
    file_count = 0
    duration = 0.0
    stats = 0
    for fname in os.listdir(dirpath):
        if not fname.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(fname)[0]
        y, sr = librosa.load(os.path.join(dirpath, fname))
        annos = load_tonal_reader(os.path.join(dirpath, f"{stem}.bin"))
        file_count += 1
        duration += len(y)/sr
        stats += len(annos)
    
    total_file_count += file_count
    total_duration += duration
    total_whistle_count += stats
    origin_data_stats[dir_name] ={
        "count": file_count,
        "total_duration": duration,
        "whistle_count": stats
    }

origin_data_stats['total'] = {
    "count": total_file_count,
    "total_duration": total_duration,
    "whistle_count": total_whistle_count
}

print("Original Data Stats:")
for category, stats in origin_data_stats.items():
    print(f"  {category}: {stats['count']} files, {stats['total_duration']/60:.2f} minutes")
    print(f"    Whistle Count: {stats['whistle_count']}")

#######################################
# Compute data stats for Refined  data
#######################################
meta = {
    "common": [
        "QX-Dc-CC0604-TAT25-060413-215524",
        "QX-Dc-CC0604-TAT25-060413-220000",
        "QX-Dc-FLIP0610-VLA-061015-165000",
        "QX-Dd-CC0604-TAT07-060406-002600",
        "Qx-Dc-CC0411-TAT11-CH2-041114-154040-s",
        "Qx-Dc-CC0411-TAT11-CH2-041114-155040-s",
        "Qx-Dc-SC03-TAT09-060516-171606",
        "Qx-Dc-SC03-TAT09-060516-173000",
        "Qx-Dd-SC03-TAT09-060516-211350",
        "Qx-Dd-SCI0608-N1-060814-150255",
        "Qx-Dd-SCI0608-N1-060815-100318",
        "Qx-Dd-SCI0608-N1-060816-142812",
        "Qx-Dd-SCI0608-Ziph-060816-151032",
        "Qx-Dd-SCI0608-Ziph-060817-082309",
        "Qx-Dd-SCI0608-Ziph-060817-125009"
    ],
    "bottlenose": [
        "Qx-Tt-SC03-TAT03-060513-212000",
        "Qx-Tt-SCI0608-N1-060814-121518",
        "Qx-Tt-SCI0608-N1-060814-123433",
        "Qx-Tt-SCI0608-Ziph-060819-072558",
        "Qx-Tt-SCI0608-Ziph-060819-074737",
        "palmyra092007FS192-070924-205305",
        "palmyra092007FS192-070924-205730",
        "palmyra092007FS192-071012-010614",
        "palmyra092007FS192-071012-012000",
        "palmyra102006-061030-230343_4"
    ],
    "melon-headed": [
        "palmyra092007FS192-070925-023000",
        "palmyra092007FS192-070927-213444",
        "palmyra092007FS192-070928-040000",
        "palmyra092007FS192-071004-032000",
        "palmyra092007FS192-071004-032342",
        "palmyra102006-061020-200922_1",
        "palmyra102006-061020-204327_4",
        "palmyra102006-061020-204454_4"
    ],
    "spinner": [
        "palmyra092007FS192-070927-224737",
        "palmyra092007FS192-070927-235000",
        "palmyra092007FS192-070928-014000",
        "palmyra092007FS192-071011-225650",
        "palmyra092007FS192-071011-232000",
        "palmyra102006-061024-225723_4",
        "palmyra102006-061103-210746_4",
        "palmyra102006-061103-212044_4",
        "palmyra102006-061103-213127_4"
    ]
}

refined_anno_dir = "data/cross/anno_refined"
refined_data_stats = {}
total_file_count = 0
total_duration = 0.0
total_whistle_count = 0
for subdir in os.listdir(origin_data_dir):
    if os.path.isfile(subdir):
        continue
    dirpath = os.path.join(origin_data_dir, subdir)
    dir_name = os.path.basename(dirpath)
    file_count = 0
    duration = 0.0
    stats = defaultdict(lambda: defaultdict(float))
    for fname in os.listdir(dirpath):
        if not fname.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(fname)[0]
        if stem not in meta[dir_name]:
            continue
        y, sr = librosa.load(os.path.join(dirpath, fname))
        anno_path = os.path.join(refined_anno_dir, f"{stem}.bin")
        annos = load_tonal_reader(anno_path)
        if stem.startswith("palmyra"):
            stats["palmyra"]["file"] += 1
            stats["palmyra"]["count"] += len(annos)
            stats["palmyra"]["duration"] += len(y)/sr
        else:
            stats["SCB"]["file"] += 1
            stats["SCB"]["count"] += len(annos)
            stats["SCB"]["duration"] += len(y)/sr
    stats['total'] = {
        "file": stats["SCB"]["file"] + stats["palmyra"]["file"],
        "count": stats["SCB"]["count"] + stats["palmyra"]["count"],
        "duration": stats["SCB"]["duration"] + stats["palmyra"]["duration"]
    }
    refined_data_stats[dir_name] = stats

refined_data_stats['total'] = {
    "palmyra": {
        "file": sum([refined_data_stats[cat]["palmyra"]["file"] for cat in meta.keys()]),
        "count": sum([refined_data_stats[cat]["palmyra"]["count"] for cat in meta.keys()]),
        "duration": sum([refined_data_stats[cat]["palmyra"]["duration"] for cat in meta.keys()]),
    },
    "SCB": {
        "file": sum([refined_data_stats[cat]["SCB"]["file"] for cat in meta.keys()]),
        "count": sum([refined_data_stats[cat]["SCB"]["count"] for cat in meta.keys()]),
        "duration": sum([refined_data_stats[cat]["SCB"]["duration"] for cat in meta.keys()]),
    },
    "total": {
        "file": sum([refined_data_stats[cat]['total']['file'] for cat in meta.keys()]),
        "count": sum([refined_data_stats[cat]['total']['count'] for cat in meta.keys()]),
        "duration": sum([refined_data_stats[cat]['total']['duration'] for cat in meta.keys()]),
    }
}

print("Refined Data Stats:")
for category, stats in refined_data_stats.items():
    print(f"  {category}: {stats['total']['count']} files, {stats['total']['duration']/60:.2f} minutes")
    print(f"    Whistle Count: {stats['total']['count']}")
    print(f"    SCB - Files: {stats['SCB']['file']}, Whistle Count: {stats['SCB']['count']}, Duration: {stats['SCB']['duration']} seconds = {stats['SCB']['duration']/60:.2f} minutes")
    print(f"    Palmyra - Files: {stats['palmyra']['file']}, Whistle Count: {stats['palmyra']['count']}, Duration: {stats['palmyra']['duration']} seconds = {stats['palmyra']['duration']/60:.2f} minutes")
