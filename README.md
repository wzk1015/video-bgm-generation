# Controllable Music Transformer

Official code for our paper *Video Background Music Generation with Controllable Music Transformer* (ACM MM 2021 Best Paper Award) 

[[Paper]](https://arxiv.org/abs/2111.08380) [[Project Page]](https://wzk1015.github.io/cmt/) [[Colab Demo]](https://colab.research.google.com/github/wzk1015/video-bgm-generation/blob/main/CMT.ipynb)



## News

[2024.12] 🚀🚀 **Check out our new [paper](https://github.com/wbs2788/VMB/tree/main) for multimodal music generation!** We propose a novel mutlimodal music generation approach with explicit text and music bridges for video-to-music, image-to-music, text-to-music, and controllable music generation tasks.

[2023.9] **Check out our new [ICCV 2023 paper](https://arxiv.org/abs/2211.11248) for video background music generation.** We provide a video and symbolic music dataset with rich annotations, an objective metric for video-music correspondence, and a benchmark model that utilizes music priors of chords, melody, and accompaniment along with video-music relations of semantic, color, and motion features.

[2022.5] **We provide a [colab notebook](https://colab.research.google.com/github/wzk1015/video-bgm-generation/blob/main/CMT.ipynb) for demo!** You can run inference code and generate background music for your input video.

[2021.10] 🏆 We won the **ACM MM 2021 [Best Paper Award](https://www.wzk.plus/award_imgs/mm.jpg)**!

## Introduction

We address the unexplored task – *video background music generation*. We first establish three rhythmic relations between video and background music. We then propose a **C**ontrollable **M**usic **T**ransformer (CMT) to achieve local and global control of the music generation process. Our proposed method does not require paired video and music data for training while generating melodious and compatible music with the given video. 

![](https://raw.githubusercontent.com/wzk1015/wzk1015.github.io/master/cmt/img/head.png)



## Directory Structure

* `src/`: code of the whole pipeline
  * `train.py`: training script, take an npz file as input music data to train the model 
  * `model.py`: code of the model
  * `gen_midi_conditional.py`: inference script, take an npz file (represents a video) as input to generate several songs
  
  * `midi2mp3.py`: script of converting midi into mp3
  
  * `src/video2npz/`: convert video into npz by extracting motion saliency and motion speed
* `dataset/`: processed dataset for training, in the format of npz
* `logs/`: logs that automatically generated during training, can be used to track the training process
* `exp/`: checkpoints, named after val loss (e.g. `loss_8_params.pt`)
* `inference/`: processed video for inference (.npz), and generated music(.mid) 




## Preparation

* Clone this repo

* Download the processed training data `lpd_5_prcem_mix_v8_10000.npz`  from [HERE](https://drive.google.com/file/d/19f_DytIbEiSDCwz8FPpScrHqmnqNtVYT/view?usp=sharing) and put it under `dataset/` 

* Download the pre-trained model `loss_8_params.pt` from [HERE](https://drive.google.com/file/d/1KvIRRm0KqTlEFDjAgs4fMtLRQBq0rBWy/view?usp=sharing) and put it under `exp/` 

* Install `ffmpeg=3.2.4` 

* Install Python3 dependencies `pip install -r py3_requirements.txt`

  * Choose the correct version of `torch` and `pytorch-fast-transformers` based on your CUDA version (see [fast-transformers repo](https://github.com/idiap/fast-transformers) and [this issue](https://github.com/wzk1015/video-bgm-generation/issues/3))

* Install `visbeat3` package: `cd src/video2npz/visbeat3; python setup.py install`

* (Optional) If you want to convert midi into mp3 with midi2audio:

  * Install fluidsynth following [this](https://github.com/FluidSynth/fluidsynth/wiki/Download)
  * Download soundfont `SGM-v2.01-Sal-Guit-Bass-V1.3.sf2` from [HERE](https://drive.google.com/file/d/1zDg0P-0rCXDl_wX4zeLcKRNmOFobq6u8/view?usp=sharing) and put it directly under this folder (`video-bgm-generation`)

  

## Training

- A quick start by using the processed data `lpd_5_prcem_mix_v8_10000.npz` (1~2 days on 8x 1080Ti GPUs):

  ```shell
  python train.py --name train_default -b 8 --gpus 0 1 2 3 4 5 6 7
  ```

* (Optional) If you want to reproduce the whole process:

  1. Download the lpd-5-cleansed dataset from [HERE](https://drive.google.com/file/d/1AzLZ4fHrcek1rVNlOC3pzxsMaNSITZsG/view?usp=sharing) and put the extracted files under `dataset/lpd_5_cleansed/`

  2. Go to `src/` and convert the pianoroll files (.npz) to midi files (~3 files / sec):

     ```shell
     python pianoroll2midi.py --in_dir ../dataset/lpd_5_cleansed/ --out_dir ../dataset/lpd_5_cleansed_midi/
     ```

  3. Convert midi files to .npz files with our proposed representation (~5 files / sec):

       ```shell
       python midi2numpy_mix.py --midi_dir ../dataset/lpd_5_cleansed_midi/ --out_name data.npz 
       ```

  4. Train the model (1~2 days on 8x 1080Ti GPUs):

      ```shell
      python train.py --name train_exp --train_data ../dataset/data.npz -b 8 --gpus 0 1 2 3 4 5 6 7
      ```

**Note:** If you want to train with another MIDI dataset, please ensure that each track belongs to one of the five instruments (Drums, Piano, Guitar, Bass, or Strings) and is named exactly with its instrument. Otherwise, you may have to change the track names in the MIDI files. You can check this with [Muspy](https://salu133445.github.io/muspy/):

```python
import muspy
midi = muspy.read_midi('xxx.mid')
print([track.name for track in midi.tracks]) # Should be like ['Drums', 'Guitar', 'Bass', 'Strings']
```



## Inference

Inference requires one GPU. You can try our [colab notebook](https://colab.research.google.com/github/wzk1015/video-bgm-generation/blob/main/CMT.ipynb) to run inference.

It is recommended to use videos *less than 2 minutes*, otherwise, it gets really slow

* Resize the video into 360p

  ```shell
  ffmpeg -i xxx.mp4 -strict -2 -vf scale=-1:360 test.mp4
  ```

* Convert input video (MP4 format) into npz

  ```shell
  cd src/video2npz
  sh video2npz.sh ../../videos/test.mp4
  ```
  
* Run model to generate `.mid` : 

  ```shell
  python gen_midi_conditional.py -f "../inference/test.npz" -c "../exp/loss_8_params.pt" -n 5
  
  # If using another training set, change `decoder_n_class` in `gen_midi_conditional` to the one in `train.py`
  ```

* Convert midi into audio

  * Get tempo of the music: 

  * ```python
     # metadata.json is generated when running `video2npz.sh`
    with open("video2npz/metadata.json") as f:
        tempo = json.load(f)['tempo']
        print("tempo:", tempo)
    ```
  * (A) Use GarageBand to convert midi into audio

    * this is **recommended** since their soundfonts are better, and no need to install fluidsynth and soundfonts
    * remember to set tempo

  * (B) Use midi2audio

    ```shell
    # Make sure you have installed fluidsynth and downloaded soundfont
    python midi2mp3.py --input ../inference/get_0.mid --output ../inference/get_0.mp3
    ```

* Combine original video and audio into video with BGM:

  ````shell
  ffmpeg -i test.mp4 -i get_0.mp3 -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 output.mp4
  
  # test.mp4: input video
  # get_0.mp3: audio file generated in the previous step
  # output.mp4: output video with BGM
  ````



## Matching Method

The matching method finds the five most matching music pieces from the music library for a given video.

```shell
python src/match.py inference/test.npz dataset/lpd_5_prcem_mix_v8_10000.npz
```



## Citation

```bibtex
@inproceedings{di2021video,
  title={Video Background Music Generation with Controllable Music Transformer},
  author={Di, Shangzhe and Jiang, Zeren and Liu, Si and Wang, Zhaokai and Zhu, Leyan and He, Zexin and Liu, Hongming and Yan, Shuicheng},
  booktitle={ACM Multimedia},
  year={2021}
}
```



## Acknowledgements

Our code is based on [Compound Word Transformer](https://github.com/YatingMusic/compound-word-transformer).

`src/video2npz/visbeat3` is built upon [haofanwang/visbeat3](https://github.com/haofanwang/visbeat3) and the original [visbeat](http://abedavis.com/visualbeat/).
