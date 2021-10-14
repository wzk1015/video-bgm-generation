echo "Note: you should use a **Python2** environment"

# extract flow magnitude into optical_flow/flow.npz
python optical_flow.py --video $1

# convert video into metadata.json with flow magnitude
python video2metadata.py --video $1

# convert metadata into .npz under `inference/`
python metadata2numpy_mix.py --video $1
