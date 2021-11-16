"""Author: Shangzhe Di (shangzhe.di@gmail.com)

Convert the pianoroll files (.npz) in the lpd-5/LPD-5-cleansed (https://salu133445.github.io/lakh-pianoroll-dataset/dataset) to midi files.

The pianoroll files are organized as:
lpd_5_cleansed
├── A
│   ├── A
│   │   ├── A
│   │   │   ├── TRAAAGR128F425B14B
│   │   │   │   └── b97c529ab9ef783a849b896816001748.npz
│   │   │   └── TRAAAZF12903CCCF6B
│   │   │       └── 05f21994c71a5f881e64f45c8d706165.npz
│   │   ├── B
│   │   │   └── TRAABXH128F42955D6
│   │   │       └── 04266ac849c1d3814dc03bbf61511b33.npz
...

The converted midi files will be organized as:
lpd_5_cleansed_midi/
├── TRDNFHP128F9324C47.mid
├── TRHIZHZ128F4295EFA.mid
├── TRRRAJP128E0793859.mid
├── TRRREEC128F9336C97.mid
├── TRRREEO128F933C62F.mid
├── TRRRERM128F429B7A0.mid
├── TRRRGET128F426655D.mid
...

"""

import pypianoroll
import argparse
import os
import os.path as osp
from tqdm import tqdm


def process_dataset(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_filename_list, out_filename_list = [], []
    for main_dir, sub_dir, filename_list in os.walk(in_dir):
        for filename in filename_list:
            if '.npz' not in filename:
                continue
            in_filename_list.append(osp.join(main_dir, filename))
            track_name = main_dir.split("/")[-1]
            out_filename_list.append(osp.join(out_dir, track_name + '.mid'))

    for i, in_filename in enumerate(tqdm(in_filename_list)):
        convert_midi(in_filename, out_filename_list[i])


def convert_midi(in_filename, out_filename):
    pianoroll = pypianoroll.load(in_filename)
    pypianoroll.write(out_filename, pianoroll)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", help='the directory of the LPD dataset', default='./lpd_5_cleansed/')
    parser.add_argument("--out_dir", help='the directory of the output midi files', default='./lpd_5_cleansed_midi/')
    args = parser.parse_args()
    process_dataset(args.in_dir, args.out_dir)
