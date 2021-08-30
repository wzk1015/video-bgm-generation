mkdir video_360p
files=$(ls video)
for filename in $files
do
    echo $filename
    ffmpeg -i video/$filename -strict -2 -vf scale=-1:360 video_360p/$filename -v quiet
done
