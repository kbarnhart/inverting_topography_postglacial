# We load our preferred compiler/MPI/python combination here:
module purge
module load intel/16.0.3
module load openmpi/1.10.2
module load python/2.7.11

#module load intel
#module load impi

# Loading these modules also initialize a few useful environment variables
# CURC_PYTHON_ROOT  : the location of our chosen  python installation
# CURC_OPENMPI_ROOT : the location of our chosen OpenMPI installation (i.e., the one compatible with Intel in this example)
# NOTE:  Whenever a module is loaded, variouse environment variables are set.  You can find/search for these using the following:
#        printenv | grep -i keyword  ; for example:  printenv | grep -i python

# Define variables for Boost's source and build directories & save the current directory
BOOST_SRC_DIR=/projects/barnhark/utils/boost_src/boost_1_53_0
BOOST_BUILD_DIR=/projects/barnhark/utils/boost/1.53
CURRENT_DIR=$PWD

# Run Boost's configure script
cd $BOOST_SRC_DIR
./bootstrap.sh --prefix=${BOOST_SRC_DIR} --with-toolset=intel-linux --with-python-root=${CURC_PYTHON_ROOT}

# In order to tell Boost to use mpi, we can edit either BOOST_SRD_DIR/project-config.jam
# or else we can add a line to the user-config.jam file in our home directory
# We use option 2


# This command CREATES the file and Appends the MPI line:
# If the file exists, this will delete the old user-config.jam file ->  take care here
echo 'using mpi : '${CURC_OPENMPI_ROOT}'/bin/mpicc ;' > ~/user-config.jam
#echo 'using mpi : '${CURC_IMPI_ROOT}'/bin64/mpicc ;' > ~/user-config.jam
# Finally, build boost (first clean, then install)
# We do this in the background.
# Monitor the build.log file to check progress.  The final line should read something like:
#  ...updated 10935 targets...
# Once the build is finished, check for error messages:  grep -i catastrophic build.log

./b2 --clean
./b2 install --prefix=${BOOST_BUILD_DIR} toolset=intel  >${BOOST_SRC_DIR}/build.log 2>&1 &


#return to the location of install_boost.sh
cd $CURRENT_DIR
~
                                                                                                          46,1          Bot
