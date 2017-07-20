PYTHON=$(shell which python3)
#DATAFILE:=$(HOME)/Work/DATA/ImagingData/2016-06-14/combined.tiff
#DATAFILE:=~/Work/DATA/ImagingData/2016-06-14/1/Trial12-ROI-1.tif
DATAFILE := ~/Work/DATA/ImagingData/test_video_stabilizer/Trial1and2.tif
CORRECTED_DATADILE:=_corrected.tif
ALL_FRAMES:=_corrected.tif.npy
CELL_FILE:=cells.npy
CELL_GRAPH:=cells_as_graph.gpickle

all :  compute_cells find_sync_cells
	@echo "All done"

compute_cells : $(CELL_FILE) ./compute_cells.py
	@echo "Finding cells in corrected tiff file"
	@echo "+ And writing a graph representing them"
	@echo "+ And generating correlation graph"

$(CELL_GRAPH) $(CELL_FILE) : $(ALL_FRAMES) ./compute_cells.py
	$(PYTHON) ./compute_cells.py -i $(ALL_FRAMES) -o $(CELL_FILE)

$(ALL_FRAMES) : $(DATAFILE) 
	$(PYTHON) ./tiffs2numpy.py -f $< -o $@

find_sync_cells: $(CELL_GRAPH)
	$(PYTHON) ./generate_community.py -c $(CELL_GRAPH)
 
clean:
	git clean -fxd

.PHONY : clean
