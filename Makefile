PYTHON=`which python`
#DATAFILE:=$(HOME)/Work/DATA/ImagingData/2016-06-14/combined.tiff
#DATAFILE:=~/Work/DATA/ImagingData/2016-06-14/1/Trial12-ROI-1.tif
DATAFILE := ~/Work/DATA/ImagingData/test_video_stabilizer/Trial1and2.tif
CORRECTED_DATADILE:=_corrected.tif
ALL_FRAMES:=_corrected.tif.npy
CELL_FILE:=cells.npy
CELL_GRAPH:=cells_as_graph.gpickle

all : stabilize_recording  compute_cells find_sync_cells
	@echo "All done"
	


stabilize_recording : $(CORRECTED_DATADILE)
	@echo "Done stabilizing image"

$(CORRECTED_DATADILE) : $(DATAFILE) 
	videostab -i $< -n 2 -o $(CORRECTED_DATADILE) -v

compute_cells : $(CELL_FILE) ./compute_cells.py
	@echo "Finding cells in corrected tiff file"
	@echo "+ And writing a graph representing them"
	@echo "+ And generating correlation graph"

$(CELL_GRAPH) $(CELL_FILE) : $(ALL_FRAMES) ./compute_cells.py
	python ./compute_cells.py -i $(ALL_FRAMES) -o $(CELL_FILE)

$(ALL_FRAMES) : $(CORRECTED_DATADILE) 
	python ./tiffs2numpy.py -f $< -o $@

find_sync_cells: $(CELL_GRAPH)
	python ./generate_community.py -c $(CELL_GRAPH)
 
clean:
	git clean -fxd

.PHONY : clean
