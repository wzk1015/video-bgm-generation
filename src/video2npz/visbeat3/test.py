import visbeat3 as vb
import os

video_dir = '../videos/'
tempos = {}
# try:
#     for file in os.listdir(video_dir):
#         if file == '.DS_Store':
#             continue
#         vlog = vb.PullVideo(name=file, source_location=video_dir+file, max_height=360)
#         # vbeats = vlog.getVisualBeatSequences(search_window=None)[0]
#         # print("vbeats are", vbeats)
#         vb.Video.getVisualTempo = vb.Video_CV.getVisualTempo
#         tempo = vlog.getVisualTempo()
#         print(file, "tempo", tempo)
#         tempos[file] = tempo
# finally:
#     print(tempos)
vlog = vb.PullVideo(source_location="../videos/wzk_vlog_beat_enhance1_track1238.mp4", max_height=360)
vb.Video.getVisualTempo = vb.Video_CV.getVisualTempo
tempo = vlog.getVisualTempo()
print(tempo)
# vbeats = vlog.getVisualBeatSequences(search_window=None)[0]
# for vbeat in vbeats:
#     print(vbeat.start, vbeat.type, vbeat.weight, vbeat.index, vbeat.unrolled_start, vbeat.direction)