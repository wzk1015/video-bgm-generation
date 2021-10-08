# CMT

code for paper Video Background Music Generation with Controllable Music Transformer (ACM MM 2021 Oral) [[arXiv]]() **#TODO**



## TODO

* fix `assert x.shape[0] == 1`

* remove `i_beat` and `n_beat` 

* `init_token` shape (related to genre and instrument)



## Directory Structure

* `src/`: code of the whole pipeline
  * `train_encoder.py`: training script, take a npz as input data to train the model 
  * `model_encoder.py`: code of the model
  * `gen_midi_conditional.py`: inference script, take several npzs (each represents a video) as input to generate several songs for each npz
  
* `src/video2npz/`: convert video into npz by extracting motion saliency and motion speed

* `lpd_dataset/`: processed LPD dataset for training, in the format of npz

* `logs/`: logs that automatically generate during training, can be used to track training status

* `exp/`: checkpoints, named after val loss (e.g. loss_13_params.pt)

* `inference/`: processed video for inference, in the format of npz




## Preparation

* clone this repo
* download `lpd_5_prcem_mix_v8_10000.npz` and put it under `lpd_dataset/`  **#TODO**

* download pretrained model `loss_8_params.pt` and put  it under `exp/` **#TODO**

* install `ffmpeg`

  ```shell
  conda install ffmpeg=4.2 -c conda-forge
  ```

* prepare a Python3 conda environment  **#TODO**

  * ```shell
    conda create -n cmt_py3 python=3.7
    pip install -r py3_requirements.txt
    ```

* prepare a Python2 conda environment (for extracting visbeat)

  * ````shell
    conda create -n cmt_py2 python=2.7
    pip install -r py2_requirements.txt
    ````

  * open `visbeat` package directory (e.g. `anaconda3/envs/cmt_py2/lib/python2.7/site-packages/visbeat`), replace the original `Video_CV.py` with `src/video2npz/Video_CV.py`



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

  ```shell
  cd src/video2npz
  sh inference.sh ../../videos/xxx.mp4
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

  

















