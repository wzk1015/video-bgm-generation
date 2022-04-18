# Controllable Music Transformer

Official code for our paper *Video Background Music Generation with Controllable Music Transformer* (ACM MM 2021 Best Paper Award) 

[[Paper]](https://arxiv.org/abs/2111.08380) [[Demos]](https://wzk1015.github.io/cmt/) [[Bibtex]](https://wzk1015.github.io/cmt/cmt.bib)



## Introduction

We address the unexplored task – *video background music generation*. We first establish three rhythmic relations between video and background music. We then propose a **C**ontrollable **M**usic **T**ransformer (CMT) to achieve local and global control of the music generation process. Our proposed method does not require paired video and music data for training while generates melodious and compatible music with the given video. 

![](https://raw.githubusercontent.com/wzk1015/wzk1015.github.io/master/cmt/img/head.png)



## Directory Structure

* `src/`: code of the whole pipeline
  * `train.py`: training script, take a npz as input music data to train the model 
  * `model.py`: code of the model
  * `gen_midi_conditional.py`: inference script, take a npz (represents a video) as input to generate several songs
  
  * `src/video2npz/`: convert video into npz by extracting motion saliency and motion speed
  
* `dataset/`: processed dataset for training, in the format of npz

* `logs/`: logs that automatically generate during training, can be used to track training process

* `exp/`: checkpoints, named after val loss (e.g. `loss_8_params.pt`)

* `inference/`: processed video for inference (.npz), and generated music(.mid) 


## Preparation

* clone this repo
* download the processed data `lpd_5_prcem_mix_v8_10000.npz`  from [HERE](https://drive.google.com/file/d/1MWnwwAdOrjC31dSy8kfyxHwv35wK0pQh/view?usp=sharing) and put it under `dataset/` 

* download the pretrained model `loss_8_params.pt` from [HERE](https://drive.google.com/file/d/1Ud2-GXEr4PbRDDe-FZJwzqqZrbbWFxM-/view?usp=sharing) and put it under `exp/` 

* install `ffmpeg=3.2.4` 

* prepare a Python3 conda environment

  * ```shell
    conda create -n mm21_py3 python=3.7
    conda activate mm21_py3
    pip install -r py3_requirements.txt
    ```
  * choose the correct version of `torch` and `pytorch-fast-transformers` based on your CUDA version (see [fast-trainsformers repo](https://github.com/idiap/fast-transformers) and [this issue](https://github.com/wzk1015/video-bgm-generation/issues/3))

  
* prepare a Python2 conda environment (for extracting visbeat)

  * ````shell
    conda create -n mm21_py2 python=2.7
    conda activate mm21_py2
    pip install -r py2_requirements.txt
    ````
    
  * open `visbeat` package directory (e.g. `anaconda3/envs/XXXX/lib/python2.7/site-packages/visbeat`), replace the original `Video_CV.py` with `src/video2npz/Video_CV.py`

## Training

**Note:** use the `mm21_py3` environment: `conda activate mm21_py3`

- A quick start using the processed data `lpd_5_prcem_mix_v8_10000.npz` (1~2 days on 8x 1080Ti GPUs):

  ```shell
  python train.py --name train_default -b 8 --gpus 0 1 2 3 4 5 6 7
  ```

* If you want to reproduce the whole process:

  1. download the lpd-5-cleansed dataset from [HERE](https://drive.google.com/uc?id=1yz0Ma-6cWTl6mhkrLnAVJ7RNzlQRypQ5) and put the extracted files under `dataset/lpd_5_cleansed/`

  2. go to `src/` and convert the pianoroll files (.npz) to midi files (~3 files / sec):

     ```shell
     python pianoroll2midi.py --in_dir ../dataset/lpd_5_cleansed/ --out_dir ../dataset/lpd_5_cleansed_midi/
     ```

  3. convert midi files to .npz files with our proposed representation (~5 files / sec):

       ```shell
       python midi2numpy_mix.py --midi_dir ../dataset/lpd_5_cleansed_midi/ --out_name data.npz 
       ```

  4. train the model (1~2 days on 8x 1080Ti GPUs):

      ```shell
      python train.py --name train_exp --train_data ../dataset/data.npz -b 8 --gpus 0 1 2 3 4 5 6 7
      ```

**Note:** If you want to train with another MIDI dataset, please ensure that each track belongs to one of the five instruments (Drums, Piano, Guitar, Bass, or Strings) and is named exactly with its instrument. You can check this with [Muspy](https://salu133445.github.io/muspy/):

```python
import muspy

midi = muspy.read_midi('xxx.mid')
print([track.name for track in midi.tracks]) # Should be like ['Drums', 'Guitar', 'Bass', 'Strings']
```

## Inference

* convert input video (MP4 format) into npz (use the `mm21_py2` environment):

  ```shell
  conda activate mm21_py2
  cd src/video2npz
  # try resizing the video if this takes a long time
  sh video2npz.sh ../../videos/xxx.mp4
  ```
  
* run model to generate `.mid` (use the `mm21_py3` environment) : 

  ```shell
  conda activate mm21_py3
  python gen_midi_conditional.py -f "../inference/xxx.npz" -c "../exp/loss_8_params.pt"
  
  # if using another training set, change `decoder_n_class` and `init_n_class` in `gen_midi_conditional` to the ones in `train.py`
  ```
  
* convert midi into audio: use GarageBand (recommended) or midi2audio 

  * set tempo to the value of  `tempo` in `video2npz/metadata.json` (generated when running `video2npz.sh`)

* combine original video and audio into video with BGM:

  ````shell
  ffmpeg -i 'xxx.mp4' -i 'yyy.mp3' -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 'zzz.mp4'
  
  # xxx.mp4: input video
  # yyy.mp3: audio file generated in the previous step
  # zzz.mp4: output video
  ````

## Matching Method

- The matching method finds the five most matching music pieces from the music library for a given video (use the `mm21_py3` environment).

  ```shell
  conda activate mm21_py3
  python src/match.py inference/xxx.npz dataset/lpd_5_prcem_mix_v8_10000.npz
  ```

## Citation

```bibtex
@inproceedings{di2021video,
  title={Video Background Music Generation with Controllable Music Transformer},
  author={Di, Shangzhe and Jiang, Zeren and Liu, Si and Wang, Zhaokai and Zhu, Leyan and He, Zexin and Liu, Hongming and Yan, Shuicheng},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={2037--2045},
  year={2021}
}
```

## Acknowledgements

Our code is based on [Compound Word Transformer](https://github.com/YatingMusic/compound-word-transformer).











