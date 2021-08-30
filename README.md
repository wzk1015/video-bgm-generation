# CMT

Unofficial code, by wzk

>  `ldp_encoder_baseline_mix_large_d_den` from 104 server



## Python Environments

### Python 3

* install dependencies according to `py3_requirements.txt` **#TODO**

### Python 2 (for extracting visbeat)

* install dependencies according to `py2_requirements.txt` **#TODO**

*  An error of `llvmlite` may appear when installing other packages (e.g.`visbeat`). If so, run the following lines instead

  ```
  pip install visbeat
  pip install llvmlite==0.20.0
  pip install visbeat --no-deps
  ```
  
  
  
  



## Directory Structure

* `src/`: code of the whole pipeline

  * `train_encoder.py`: training script, take a npz as input data to train the model 
  * `model_encoder.py`: code of the model
  * `gen_midi_conditional.py`: inference script, take several npzs (each represents a video) as input to generate several songs for each npz
  * `numpy2midi_mix.py`: convert numpy array into midis, used by `gen_midi_conditional`

  * `utils.py`: some useful functions
  * `midi2numpy_mix_util.py`: convert midi into numpy array, used to produce training data
  * `match.py`: calculate matchness between video and music (density and strength)
  * `metadata_v2.json`: an example of metadata extracted from each video, including duration, tempo, flow_magnitude_per_bar and visbeats.
  * `dictionary_mix.py`: a preset dictionary of compound word representation

* `src/video2npz/`: convert video into npz by extracting motion saliency and motion speed

* `lpd_dataset/`: processed LPD dataset for training, in the format of npz

* `logs/`: logs that automatically generate during training, can be used to track training status

* `exp/`: checkpoints, named after val loss (e.g. loss_13_params.pt)

* `inference_npz/`: processed video for inference, in the format of npz




## Preparation

* clone this repo
* download `lpd_5_ccdepr_mix_v4_10000.npz` and put it under `lpd_dataset/`

* download pretrained model `loss_13_params.pt` and put  it under `exp/`



## Training

* convert training data from midi into npz **#TODO**

  * ```shell
    python midi2numpy_mix.py --midi wzk/wzk_vlog_beat_enhance1_track1238.mid --visualize --video wzk.mp4
    ```

  * `--visualize` and `--video`: used to provide figures like Figure 2 in the paper. Put `metadata_v2.json` under the same directory first.

* modify `path_train_data` in `train_encoder.py`

  * default: `lpd_5_ccdepr_mix_v4_10000.npz`

* run `python3 train_encoder.py -n XXX`, where XXX is the name of the experiment (will be the name of the log file & the checkpoints directory)
  
  * if XXX is `debug`, checkpoints will not be saved
  * change LR, batch size by adding arguments (see `parser` in `train_encoder.py`)
  * change epochs or other model hyperparameters by modifying the source .py files
  * continue training by adding `-p` argument to load model from given path

## Inference

* convert video into npz **#TODO**
  * run `python video2metadata.py --process_all --video_dir xxx/` to get file `metadata.json` (You should use a python2 environment)
  * run `python metadata2numpy_mix.py --video_dir xxx/ --metadata xxx/metadata.json`  to get npz file
* put npz under `inference/`
* modify `path_saved_ckpt` in `gen_midi_conditional.py` 
  * default: pretrained `loss_13_params.pt`
* run `python3 gen_midi_conditional.py`

















