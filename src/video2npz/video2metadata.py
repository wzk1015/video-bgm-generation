#/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib

matplotlib.use('Agg')
import visbeat3 as vb
import os
import os.path as osp
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def makedirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def frange(start, stop, step=1.0):
    while start < stop:
        yield start
        start += step


def process_all_videos(args):
    out_json = {}
    for i, video_name in enumerate(os.listdir(args.video_dir)):
        if '.mp4' not in video_name:
            continue
        print('%d/%d: %s' % (i, len(os.listdir(args.video_dir)), video_name))
        metadata = process_video(video_name, args)
        out_json[video_name] = metadata

    json_str = json.dumps(out_json, indent=4)
    with open(osp.join(args.video_dir, 'metadata.json'), 'w') as f:
        f.write(json_str)


def process_video(video_path, args):
    figsize = (32, 4)
    dpi = 200
    xrange = (0, 95)
    x_major_locator = MultipleLocator(2)

    vb.Video.getVisualTempo = vb.Video_CV.getVisualTempo

    video = os.path.basename(video_path)
    vlog = vb.PullVideo(name=video, source_location=osp.join(video_path), max_height=360)
    vbeats = vlog.getVisualBeatSequences(search_window=None)[0]

    tempo, beats = vlog.getVisualTempo()
    print("Tempo is", tempo)
    vbeats_list = []
    for vbeat in vbeats:
        i_beat = round(vbeat.start / 60 * tempo * 4)
        vbeat_dict = {
            'start_time': vbeat.start,
            'bar'       : int(i_beat // 16),
            'tick'      : int(i_beat % 16),
            'weight'    : vbeat.weight
        }
        if vbeat_dict['tick'] % args.resolution == 0:  # only select vbeat that lands on the xth tick
            vbeats_list.append(vbeat_dict)
    print('%d / %d vbeats selected' % (len(vbeats_list), len(vbeats)))

    npz = np.load("flow/" + video.replace('.mp4', '.npz'), allow_pickle=True)
    print(npz.keys())
    flow_magnitude_list = npz['flow']
    fps = round(vlog.n_frames() / float(vlog.getDuration()))
    fpb = int(round(fps * 4 * 60 / tempo))  # frame per bar

    fmpb = []  # flow magnitude per bar
    temp = np.zeros((len(flow_magnitude_list)))
    for i in range(0, len(flow_magnitude_list), fpb):
        mean_flow = np.mean(flow_magnitude_list[i: min(i + fpb, len(flow_magnitude_list))])
        fmpb.append(float(mean_flow))
        temp[i: min(i + fpb, len(flow_magnitude_list))] = mean_flow

    if args.visualize:
        makedirs('image')

        height = vlog.getFrame(0).shape[0]
        thumbnails = [vlog.getFrameFromTime(t)[:, :int(height * 2.5 / 10), :] for t in list(frange(25, 35, 1))]
        thumbnails = np.concatenate(thumbnails, axis=1)
        cv2.cvtColor(thumbnails, cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join('image', video + '_thumbnails_1' + '.png'), thumbnails)

        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=figsize, dpi=dpi)
        plt.subplots_adjust(bottom=0.15)

        x2_time = [float(item) / fps for item in list(range(len(flow_magnitude_list)))]
        plt.plot(x2_time[::3], flow_magnitude_list[::3], '-', color='#fff056', alpha=0.75, label="Per Frame")
        for i, fm in enumerate(fmpb):
            x_frame = [i * fpb, (i + 1) * fpb - 1]
            x_time = [x / fps for x in x_frame]
            y_fm = [fm, fm]
            if i == 0:
                plt.plot(x_time, y_fm, 'r-', label='Per Bar', lw=3)
            else:
                plt.plot(x_time, y_fm, 'r-', lw=3)
        if xrange is not None:
            plt.xlim(xrange)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlabel('Time (s)')
        plt.ylabel('Optical Flow Magnitude')
        plt.legend(loc="upper left")
        plt.savefig(osp.join('image', video + '_flow' + '.eps'), format='eps', transparent=True)
        plt.savefig(osp.join('image', video + '_flow' + '.png'), format='png', transparent=True)

        vlog.printVisualBeatSequences(figsize=figsize, save_path=osp.join('image', video + '_visbeat' + '.eps'),
                                      xrange=xrange, x_major_locator=x_major_locator)

    return {
        'duration'              : vlog.getDuration(),
        'tempo'                 : tempo,
        'vbeats'                : vbeats_list,
        'flow_magnitude_per_bar': fmpb,
    }


if __name__ == '__main__':
    # vb.SetAssetsDir('.' + os.sep + 'VisBeatAssets' + os.sep)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../../videos/final_640.mp4')
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--resolution', type=int, default=1)
    args = parser.parse_args()

    metadata = process_video(args.video, args)
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
    print("saved to metadata.json")