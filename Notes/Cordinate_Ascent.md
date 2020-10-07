#### #20200405
- Two-reward paradigm works fine on `CHX`, `ESM` and `FMX` beamlines.
- Two-reward paradigm is unable to run on `CSX` beamline. The reason is  for 
the initial values, the output image is blank. 

#### #20200420
- Use an unified loss function for all parameters.
- Coordinate Ascent method works on `FMX`, `FMX-2`, `ESM`, `CHX`, `HXN`, `SMI`.
- Dont't work on `CSX` because the image is all zero if the parameters are not appropriate.
- Have less satisfied result on `SRX`.
- Should we tune the alpha parameters of the reward for different expriments?

Questions:
- In `HXN`, `SRX` file, there are several generated `.dat` files, should we consider all of them? Some 
files lose the dimension information.

#### #20200510
Changes:
- When computing the complexity, only concerns the tuned parameter lists.

#### #20200910
Changes:
- Update the framework of the previous Coordinate Ascent function into a more reusable 
class framework. So now we can gradually add more optimizers into the package.