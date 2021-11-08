import os


def resize_video(video_name, in_dir='video', out_dir='video_360p', max_height=360):
    command = 'ffmpeg -i %s -strict -2 -vf scale=-1:%d %s -v quiet' % (
        os.path.join(in_dir, video_name),
        max_height,
        os.path.join(out_dir, video_name)
    )
    print(command)
    os.system(command)
    return os.path.join(out_dir, video_name)