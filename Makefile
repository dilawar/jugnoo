PYTHON=`which python`
DATADIR:=$(HOME)/Work/DATA/g5_136/test/

all : all_frames.npy  cells.png.npy result.png
	@echo "All done"

all_frames.npy : tiffs2numpy.py
	@echo "Processind directory $(DATADIR)"
	$(PYTHON) $< -d $(DATADIR) -o $@

cells.png.npy : all_frames.npy
	$(PYTHON) ./compute_cells.py $< 

result.png : cells.png.npy all_frames.npy 
	$(PYTHON) ./cluster_cells.py $?

clean : 
	rm -rf *.png 

.PHONY : clean
