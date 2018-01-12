# MeanShiftGPU

This C and OpenCL code demonstrates GPU speedup when performing a MeanShift Clustering algorighm.

## Installation

### Dependencies 

MeanShiftGPU depends on [gsl](https://www.gnu.org/software/gsl/) (GNU Scientific Library).
Intall via your OS's package manager on Linux, or via the [homebrew](https://brew.sh/) package manager for OSX:

```
brew install gsl
```
If you don't have Homebrew, you can install it by

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Instructions for Windows can be found [here](http://www2.lawrence.edu/fast/GREGGJ/CMSC210/gsl/gsl.html).


Of course, we also require the OpenCL framework, which comes with modern versions of OSX. Try these instructions for [Linux](https://wiki.tiker.net/OpenCLHowTo/), 
and these for [Windows](https://streamhpc.com/blog/2015-03-16/how-to-install-opencl-on-windows/).

In order to view the results, [GNUplot](http://www.gnuplot.info/) is optionally required, but recommended. With homebrew
```
brew install gnuplot --with-qt
```
or set up as you prefer.

### Building

Git clone. In the ```MeanShiftGPU``` folder, build the applicaton via:

```
make && make clean
```

You may get a couple of warnings, but hopefully no errors!

The ```clean``` call will move the binary and OpenCL headers to the ```./bin``` subfolder.

## Usage

```
./bin/MeanSHiftGPU [seed]
```

where *seed* is the integer seed used by the random number generator.

The program will write two files ```data.temp``` and ```result.temp```. If you do not have GNUplot, you may use these files with your preferred plotting software.
Their formats are

* ```data.temp```: x, y, cluster-identity
* ```result.temp```: x, y, final-x-shift, final-y-shift, cluster-id, estimated-cluster-id, estimated-cluster-id2

where _estimated-cluster-id2_ ignores estimated-clusters with fewer members than 0.75 of the actual cluster size.
