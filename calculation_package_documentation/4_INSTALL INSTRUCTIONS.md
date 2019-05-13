# SOFTWARE INSTALLATION INSTRUCTIONS

Katherine Barnhart
March 2017 - March 2018

The following instructions describe how to install dependencies and python
modules used for the West Valley Erosion Working Group Study 3 Modeling Effort.

# 0. Computational requirements
This calculation package was designed to run on a Red Hat Enterprise Linux 7
supercomputer running the slurm submission protocol. It was run on the Summit
heterogeneous supercomputing cluster.

(https://www.rc.colorado.edu/resources/compute/summit).

Undertaking the development, testing, final model evaluations, and
post-processing analyses took 1.36 Million Core Hours.

Re-running the entire modeling and analysis suite will create approximately
5 TB of files.

The package includes a number of files and example jupyter notebooks that are
appropriate to run on an individual's laptop or desktop.

# 1. Install dependencies

a. Git

Many platforms come with git already installed.
If your platform does not have git installed, first install it.

The following URL has information on how to set up git.

https://help.github.com/articles/set-up-git/

b. Curl

Curl is a command line tool for transferring data using URLs. This install
guide uses Curl to download a miniconda distribution.

c. Anaconda Python Distribution

Next, install the Anaconda python distribution.

First option is to install the full distribution from Anaconda. The following
URL will provide download options from Anaconda.

https://www.continuum.io/downloads

A second option that is sufficient if you are space limited is to install the
miniconda distribution. This is smaller version of Anaconda. After installing
miniconda you will need to set paths and install a python version. We
recommend using either Python 2.7 or Python 3.5.
The following example lines are from installing on MacOSX with bash
and curl. Depending on your OS and shell, change as needed.

Here we show options for MacOSX and for Linux.

```
$ curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh > miniconda.sh
$ curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
```

Next, install miniconda. $HOME here is the directory in which you would like
the conda installation to be located.

```
$ bash ./miniconda.sh -b -f -p $HOME/
```

Next, edit your .bash_profile file to include the following line.
This will ensure that your computer can see the conda tool.
```
$ export PATH=$HOME/conda/bin:$PATH
```
Finally run the following two lines to update conda and install Python 2.7.
Most of the packages used by this project are tested on both Python 2.7 and
Python 3.5. However, not all of them are actively tested on Python 3.5 and
thus we recommend using Python 2.7.
```
$ conda update --all
$ conda install python=2.7
```
d. Landlab

Once git, conda, and python are installed, you can install Landlab. Navigate
to the folder where you'd like Landlab to be located and clone it from git.
```
$ git clone https://github.com/landlab/landlab
$ cd landlab
$ conda install --file=requirements.txt
$ python setup.py develop
```

This work used landlab version `v1.2.0+481.ga4b5873`.

Landlab should now be installed. To test Landlab do the following. A lot of
text will print to the console. If "OK" is printed on one of the last lines,
you will know that Landlab has passed all tests.
```
$ cd scripts
$ python test-installed-landlab.py
```
e. Dakota
Dakota is available from Sandia National Labs

https://dakota.sandia.gov/download.html

Download the correct version for your platform and follow the install
instructions.

https://dakota.sandia.gov/quickstart.html

You will need to make sure that your paths are set correctly so that Dakota
can be run from the command line. Follow the instructions below to set
environment variables for Dakota.

https://dakota.sandia.gov/content/set-environment

On the Summit compute resource I built Dakota from source. General
instructions for compiling Dakota from source are available here:

https://dakota.sandia.gov/content/build-compile-source-code

It is important to make sure that you have build Dakota with the QUESO package
enabled.

The following steps were successfully used to install Dakota on the CU Summit
Supercomputer.

On Summit, building Dakota from source requires installing Boost 1.53. This
is the recommended Boost version and it is older than the version available
on Summit for general use. First download and extract Boost.

First, create a folder where you will install Boost and Dakota. I used:
```
/projects/barnhark/utils/
```
I will refer to this filepath as `$UTL_FP`

Next, place the three files:

- `install_boost.sh`

- `install_dakota.sh`

- `dakota_cmake_openmpi_intel_boost1.53`

into this folder. Next, navigate to this directory and download boost.
```
$ curl -L https://sourceforge.net/projects/boost/files/boost/1.53.0/boost_1_53_0.tar.gz > boost_1_53_0.tar.gz
$ tar xzvf boost_1_53_0.tar.gz
```
Remove the tarball.
```
$ rm boost_1_53_0.tar.gz
```
I created a director `boost_src` within `$UTL_FP` to store all versions of Boost
source code.
```
$ mkdir boost_src
$ mv boost_1_53_0/ boost_src/
```
Next, build boost. Configure `install_boost.sh` to have your necessary filepaths
and run from within `$UTL_FP`
```
$ source install_boost.sh
```
Next, download and extract Dakota to `$UTL_FP`.
```
$ cd /path/to/Dakota/install/directory
$ curl https://dakota.sandia.gov/sites/default/files/distributions/public/dakota-6.5-release-public-src.tar.gz > dakota-6.5-release-public-src.tar.gz
$ tar xzvf dakota-6.5-release-public-src.tar.gz
```
Remove the tarball.
```
$ rm dakota-6.5-release-public-src.tar.gz
```
I created a directory `dakota_src` within `$UTL_FP` to store all versions of Dakota
source code.
```
$ mkdir dakota_src
$ mv dakota-6.5.0.src/ dakota_src/
```
Next, build Dakota. Configure `install_dakota.sh` to have your
necessary filepaths and run from within `$UTL_FP`
```
$ source install_dakota.sh
```
This took about 30 minutes on Summit.

Next, set paths for Dakota. Edit `.bash_rc` or `.bash_profile` so the path
includes Dakota
```
INSTALL_DIR=PUT_THE_DAKOTA_INSTALL_PATH_HERE
export PATH=$INSTALL_DIR/bin:$INSTALL_DIR/test:$INSTALL_DIR/gui:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INSTALL_DIR/lib:$INSTALL_DIR/bin
```
Test by typing:
```
$ dakota -v
```
I get
```
Dakota version 6.5 released Nov 13 2017.
Repository revision f928f89 (2016-11-10) built Nov 13 2017 15:03:05.
```
Your install will have a different build day.

Next test by running the recommended Dakota described here:
https://dakota.sandia.gov/content/test-built-dakota

f. Dakotathon Python Module

Some of the work with Dakota used the CSDMS Dakotathon tool.

https://csdms.colorado.edu/wiki/Model:Dakotathon

To install, do the following:

```
$ git clone https://github.com/csdms/dakota.git
$ cd dakota
$ python setup.py install
```

g. R software

Download and compile R software. This work used

R version 3.4.3 (2017-11-30) -- "Kite-Eating Tree"

which was already provided on CU's Summit platform.

R is avalable from https://www.r-project.org

Additionally the following R libraries (versions listed) were used:

- dplyr_0.7.4
- forcats_0.2.0
- GGally_1.3.2  
- ggplot2_2.2.1      
- gridExtra_2.3
- latex2exp_0.4.0
- MASS_7.3-48       
- ncdf4_1.16
- plyr_1.8.4
- png_0.1-7         
- purrr_0.2.4       
- readr_1.1.1       
- reshape2_1.4.3
- rgeos_0.3-26     
- stargazer_5.2.1   
- stringr_1.2.0        
- viridis_0.4.0     
- viridisLite_0.2.0
- tibble_1.4.1
- tidyverse_1.2.1  
- tidyr_0.7.2       


Note that rgeos requires the GEOS framework available at: https://trac.osgeo.org/geos.

# 2. Put the calculation package in the correct location.

This calculation package is designed to be run from the location `/work/WVDP_EWG_STUDY3/`

It often used relative file paths and built in functions of R and python that
permit construction of file paths correctly across platforms, but it does not
always do this. It is recommended that it is run from the correct location and
on a Linux platform.

# 3. Compile modules within *study3py*.

This calculation package is designed to be run from the location `/work/WVDP_EWG_STUDY3/`

Such that the `study3py` folder is located at `/work/WVDP_EWG_STUDY3/study3py`

If it is not located in this directory many file paths within the package will need to change.

First, navigate to the `study3py` folder.

a. Compile Erosion Model Suite

Next we install the `erosion_model` python module:

```
cd erosion_modeling_suite
python setup.py install
```

b. Compile Metric Calculator

Finally install the `metric_calculator`

```
cd ../metric_and_objective_function_calculation
python setup.py install
```
