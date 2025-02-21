from pathlib import Path

from whistle_prompter import utils

if __name__ == "__main__":
    with open('data/meta.json') as f:
        import json
        meta = json.load(f)
    for split, stems in meta['data'].items():
        for stem in stems:
            # read binary annotation file for each audio
            bin_file = Path(f'data/raw_anno/{stem}.bin')
            annos = utils.load_annotation(bin_file)
            traj0 = annos[0]
            print(traj0.shape)
            dense_traj0 = utils.get_dense_annotation(traj0)
            print(dense_traj0.shape)
            dense_traj0_pix = utils.tf_to_pix(dense_traj0)
            print(dense_traj0_pix.shape)
            dense_traj0_poly = utils.polyline_to_polygon(dense_traj0_pix)
            print(dense_traj0_poly)
