module purge
module load intel/16.0.3
module load openmpi/1.10.2
#module load mkl/11.3.3
module load cmake/3.5.2
#module load gsl/2.1
#module load perl/5.24.0
#module load intel
#module load impi
module load perl
#module load cmake
module load mkl
module load gsl

#NOTE:  DO NOT load the boost module.  It will cause conflicts during the build process.

CDIR=$PWD
UTIL_DIR=/projects/barnhark/utils
DAK_SRC=${UTIL_DIR}/dakota_src/dakota-6.5.0.src
DAK_BUILD=${UTIL_DIR}/dakota/6.5.0b_with_queso
DAK_INSTALL=${UTIL_DIR}/dakota/6.5.0_with_queso
DAK_CMAKE_FILE=${UTIL_DIR}/dakota_cmake_openmpi_intel_boost1.53_with_queso

#-DDAKOTA_HAVE_MPI:BOOL=TRUE

#This script does a complete build of Dakota
mkdir -p $DAK_BUILD
mkdir -p $DAK_INSTALL
cd $DAK_BUILD
cmake -DCMAKE_INSTALL_PREFIX=${DAK_INSTALL} -C ${DAK_CMAKE_FILE} ${DAK_SRC}
make -j 4
make install
cd $CDIR
                                                                    5,1           All
