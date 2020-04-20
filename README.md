# SRW_PROP
This project contains several tools to optimize the propagation parameters
of SRW simulator.


## Configuration
### Prerequisites
- SRW: `git clone https://github.com/ochubar/SRW.git`
- Skimage:

### Add `SRW` as Python Package
Before testing the propagation parameter optimization, you should 
add `SRW` as a package of your Python. This can be achieved by 
copying the `SRW.pth` into the site-packages folder of your Python
and change the path as `PATH_TO_SRW\env\work\srw_python` in `SRW.pth` to the path of the SRW package after
the git.

## #20200405
- Coordinate ascent method works fine on `CHX`, `ESM` and `FMX` beamlines.
- Coordinate ascent method unable to run on `CSX` beamline. The reason is  for 
the initial values, the output image is blank. 