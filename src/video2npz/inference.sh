# extract flow magnitude into optical_flow/flow.npz
conda activate cmt_py3
python optical_flow.py --video $1

# convert video into metadata.json with flow magnitude
conda activate cmt_py2
python video2metadata.py --video $1

# convert metadata into .npz under `inference/`
conda activate cmt_py3
python metadata2numpy_mix.py --video $1