# CMT

Unofficial code, by wzk

## TODO

* fix `assert x.shape[0] == 1`

* remove `i_beat` and `n_beat` 

* `init_token` shape 

## Python Environments

### FFMPEG

```shell
conda install ffmpeg=4.2 -c conda-forge
```



### Python 3

install dependencies according to `py3_requirements.txt` **#TODO**

### Python 2 (for extracting visbeat)

```shell
pip install -r py2_requirements.txt
```

open `visbeat` package directory (e.g. `anaconda3/envs/xxx/lib/python2.7/site-packages/visbeat`), **replace the original `Video_CV.py` with `src/video2npz/Video_CV.py`**





## Directory Structure

* `src/`: code of the whole pipeline

  * `train_encoder.py`: training script, take a npz as input data to train the model 
  * `model_encoder.py`: code of the model
  * `gen_midi_conditional.py`: inference script, take several npzs (each represents a video) as input to generate several songs for each npz
  * `numpy2midi_mix.py`: convert numpy array into midis, used by `gen_midi_conditional`

  * `utils.py`: some useful functions
  * `midi2numpy_mix_util.py`: convert midi into numpy array, used to produce training data
  * `match.py`: calculate matchness between video and music from training set (density and strength) and select the closet one as output (see Equation 16 in the paper)
  * `metadata_v2.json`: an example of metadata extracted from each video, including duration, tempo, flow_magnitude_per_bar and visbeats.
  * `dictionary_mix.py`: a preset dictionary of compound word representation

* `src/video2npz/`: convert video into npz by extracting motion saliency and motion speed

* `lpd_dataset/`: processed LPD dataset for training, in the format of npz

* `logs/`: logs that automatically generate during training, can be used to track training status

* `exp/`: checkpoints, named after val loss (e.g. loss_13_params.pt)

* `inference/`: processed video for inference, in the format of npz




## Preparation

* clone this repo
* download `lpd_5_prcem_mix_v8_10000.npz` and put it under `lpd_dataset/`

* download pretrained model `loss_8_params.pt` and put  it under `exp/`



## Training

* If you want to use another training set:  convert training data from midi into npz

  ```shell
  python midi2numpy_mix.py --midi_dir /PATH/TO/MIDIS/ 
  ```

  

* train the model

  ```shell
  python train_encoder.py -n XXX
  
  # -n XXX: the name of the experiment, will be the name of the log file & the checkpoints directory. if XXX is 'debug', checkpoints will not be saved
  # -l (--lr): initial learning rate
  # -b (--batch_size): batch size
  # -p (--path): if used, load model checkpoint from the given path
  # -e (--epochs): number of epochs in training
  # -t (--train_data): path of the training data (.npz file) 
  # other model hyperparameters: modify the source .py files
  ```



## Inference

* convert video into npz 

  put video under `../../videos`

  ```shell
  cd video2npz
  
  # extract flow magnitude into optical_flow/flow.npz
  # use Python3 env
  python optical_flow.py --video ../../videos/xxx.mp4
  
  # convert video into metadata.json with flow magnitude
  # use **Python2** env
  python video2metadata.py --video ../../videos/xxx.mp4
  
  # convert metadata into .npz under `inference/`
  # use Python3 env
  python metadata2numpy_mix.py --name xxx.mp4
  ```

  

* run model to generate `.mid` : 

  ```shell
  python gen_midi_conditional.py -f "../inference/xxx.npz" -c "../exp/loss_8_params.pt"
  
  # -c (--ckpt): checkpoints to be loaded
  # -f (--files): input npz file
  ```

  

* convert midi into audio (e.g. `.m4a`): use GarageBand (recommended) or midi2audio 

  * if using GarageBand, change tempo to the value of  `tempo` in `metadata.json` 

  

* combine original video and audio into video with BGM

  ````shell
  ffmpeg -i 'xxx.mp4' -i 'yyy.m4a' -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 'zzz.mp4'
  
  # xxx.mp4: input video
  # yyy.m4a: audio file generated in the previous step
  # zzz.mp4: output video
  ````

  

















