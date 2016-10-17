PYTHON=`which python`
DATADIR:=$(HOME)/Work/DATA/g5_136/test/
DATAFILE:=$(HOME)/Work/DATA/ImagingData/2016-06-14/1/Trial10-ROI-1.tif
CORRECTED_DATADILE:=_corrected.tif
ALL_FRAMES:=_corrected.tif.npy
CELL_FILE:=cells.npy
CORRELATION_GRAPH:=correlation_graph.pickle

all : stabilize_recording  compute_cells find_sync_cells
	@echo "All done"
	


stabilize_recording : $(CORRECTED_DATADILE)
	@echo "Done stabilizing image"

$(CORRECTED_DATADILE) : $(DATAFILE) 
	videostab -i $< -n 4 -o $(CORRECTED_DATADILE)

compute_cells : $(CELL_FILE) ./compute_cells.py
	@echo "Finding cells in corrected tiff file"

$(CELL_FILE) : $(ALL_FRAMES)
	python ./compute_cells.py -i $(ALL_FRAMES) -o $(CELL_FILE)

$(ALL_FRAMES) : $(CORRECTED_DATADILE) 
	python ./tiffs2numpy.py -f $< -o $@

correlation_graph: $(CORRELATION_GRAPH)
	@echo "Genearted correlation graph"


$(CORRELATION_GRAPH) : $(ALL_FRAMES) $(CELL_FILE) ./generate_correlation_graph.py 
	@echo "Generating correlation graph"
	python ./generate_correlation_graph.py --cells $(CELL_FILE) \
	    --frames $(ALL_FRAMES) -o $(CORRELATION_GRAPH)

find_sync_cells : $(CORRELATION_GRAPH) ./generate_community.py
	@echo "Generating synchornized cells"
	$(PYTHON) ./generate_community.py --community $(CORRELATION_GRAPH) \
	    --frames $(ALL_FRAMES)

## 
## #result.png : ./correlation_graph.pickle cells.png.npy all_frames.npy 
## 	#$(PYTHON) ./generate_community.py --community ./community_graph.pickle
## 
## #correlation_graph.pickle : ./generate_correlation_graph.py 
## 	#$(PYTHON) ./generate_correlation_graph.py --cells ./cells.png.npy \
## 	    #--frames ./all_frames.npy
## 
## #clean : 
## 	#rm -rf *.png 
 
.PHONY : clean
