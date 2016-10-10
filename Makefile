PYTHON=`which python`


all : all_frames.npy  cells.png.npy result.png
	@echo "All done"

all_frames.npy : tiffs2numpy.py
	$(PYTHON) $< -d ~/Work/DATA/g5_136/test/ -o $@

cells.png.npy : all_frames.npy
	$(PYTHON) ./compute_cells.py $< 

result.png : cells.png.npy all_frames.npy 
	$(PYTHON) ./cluster_cells.py $?
