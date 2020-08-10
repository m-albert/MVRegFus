# TODO

### User interface
- GUI or napari plugin
- limit output to informative messages
- informative error message when registration between pair of views fails, and possibly save input images to temp dir
- add ipython notebook for debugging (napari?)


###Performance
- stream fused blocks directly to file
- perform blockwise transformations to handle large target volumes
- optimize weight calculation (downscaling seems to take long)


### Registration
- be tolerant to wrong registrations
  - build graph for registrations and resolve shortest paths
  - build candidate pairs from metadata (e.g. group in y, define weights by difference of angle, distance between middle of stacks)