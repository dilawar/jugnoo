PYTHON=`which python`
DATADIR:=$(HOME)/Work/DATA/g5_136/test/

all : all_frames.npy  cells.png.npy result.png
	@echo "All done"

all_frames.npy : tiffs2numpy.py
	@echo "Processind directory $(DATADIR)"
	$(PYTHON) $< -d $(DATADIR) -o $@

cells.png.npy : all_frames.npy
	$(PYTHON) ./compute_cells.py $< 

result.png : ./correlation_graph.pickle cells.png.npy all_frames.npy 
	$(PYTHON) ./generate_community.py --community ./community_graph.pickle

correlation_graph.pickle : ./generate_correlation_graph.py 
	$(PYTHON) ./generate_correlation_graph.py --cells ./cells.png.npy \
	    --frames ./all_frames.npy

clean : 
	rm -rf *.png 

.PHONY : clean
