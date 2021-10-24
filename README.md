# CMT

Code for paper Video Background Music Generation with Controllable Music Transformer (ACM MM 2021 Best Paper Award) 

[[Paper]](https://raw.githubusercontent.com/wzk1015/wzk1015.github.io/master/cmt/paper.pdf) [[Site]](https://wzk1015.github.io/cmt/)





## Directory Structure

* `src/`: code of the whole pipeline
  * `train.py`: training script, take a npz as input music data to train the model 
  * `model.py`: code of the model
  * `gen_midi_conditional.py`: inference script, take a npz (represents a video) as input to generate several songs
  
  * `src/video2npz/`: convert video into npz by extracting motion saliency and motion speed
  
* `dataset/`: processed dataset for training, in the format of npz

* `logs/`: logs that automatically generate during training, can be used to track training process

* `exp/`: checkpoints, named after val loss (e.g. loss_13_params.pt)

* `inference/`: processed video for inference (.npz), and generated music(.mid) 




## Preparation

* clone this repo
* download `lpd_5_prcem_mix_v8_10000.npz`  from [HERE](https://drive.google.com/file/d/1Ud2-GXEr4PbRDDe-FZJwzqqZrbbWFxM-/view?usp=sharing) and put it under `dataset/` 

* download pretrained model `loss_8_params.pt` from [HERE](https://drive.google.com/file/d/1MWnwwAdOrjC31dSy8kfyxHwv35wK0pQh/view?usp=sharing) and put  it under `exp/` 

* install `ffmpeg=3.2.4` 

* prepare a Python3 conda environment

  ```shell
  pip install -r py3_requirements.txt
  ```
  
* prepare a Python2 conda environment (for extracting visbeat)

  * ````shell
    pip install -r py2_requirements.txt
    ````
    
  * open `visbeat` package directory (e.g. `anaconda3/envs/XXXX/lib/python2.7/site-packages/visbeat`), replace the original `Video_CV.py` with `src/video2npz/Video_CV.py`



## Training

* If you want to use another training set:  convert training data from midi into npz under `dataset/`

  ```shell
  python midi2numpy_mix.py --midi_dir /PATH/TO/MIDIS/ --out_name data.npz 
  ```

* train the model

  ```shell
  python train.py -n XXX -g 0 1 2 3
  
  # -n XXX: the name of the experiment, will be the name of the log file & the checkpoints directory. if XXX is 'debug', checkpoints will not be saved
  # -l (--lr): initial learning rate
  # -b (--batch_size): batch size
  # -p (--path): if used, load model checkpoint from the given path
  # -e (--epochs): number of epochs in training
  # -t (--train_data): path of the training data (.npz file) 
  # -g (--gpus): ids of gpu
  # other model hyperparameters: modify the source .py files
  ```



## Inference

* convert input video (MP4 format) into npz (use the **Python2** environment)

  ```shell
  cd src/video2npz
  sh video2npz.sh ../../videos/xxx.mp4
  ```
  
  * try resizing the video if this takes a long time
  
  
  
* run model to generate `.mid` : 

  ```shell
  python gen_midi_conditional.py -f "../inference/xxx.npz" -c "../exp/loss_8_params.pt"
  
  # -c: checkpoints to be loaded
  # -f: input npz file
  # -g: id of gpu (only one gpu is needed for inference) 
  ```

  * if using another training set, change `decoder_n_class` in `gen_midi_conditional` to the `decoder_n_class` in `train.py`

  

* convert midi into audio: use GarageBand (recommended) or midi2audio 

  * set tempo to the value of  `tempo` in `video2npz/metadata.json` 

  

* combine original video and audio into video with BGM

  ````shell
  ffmpeg -i 'xxx.mp4' -i 'yyy.mp3' -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 'zzz.mp4'
  
  # xxx.mp4: input video
  # yyy.mp3: audio file generated in the previous step
  # zzz.mp4: output video
  ````

  

















