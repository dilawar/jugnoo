PYTHON=`which python`


all : all_frames.npy 
	@echo "All done"

all_frames.npy : tiffs2numpy.py
	$(PYTHON) $< -d ~/Work/DATA/g5_136/test/ -o $@
