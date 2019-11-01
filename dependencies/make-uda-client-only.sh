#!/bin/bash

UDA_TAG=2.2.6
UDA_RELEASE_DIR=$PWD/uda-release-2.2.6
UDA_INSTALL_DIR=$PWD/uda-install-2.2.6

# SWIG location, needed on Freia but not if swig is system installed
SWIG_DIR=/common/projects/UDA/swig/bin/swig

# Clone UDA repo
git clone git@git.ccfe.ac.uk:MAST-U/UDA.git -b $UDA_TAG $UDA_RELEASE_DIR || die "failed to clone UDA repo"

# configure, make and install into $UDA_INSTALL_DIR
cd $UDA_RELEASE_DIR
CC=gcc CXX=g++ cmake3 -Bbuild -H. -DCMAKE_BUILD_TYPE=Debug -DTARGET_TYPE=MAST \
   -DCMAKE_INSTALL_PREFIX=$UDA_INSTALL_DIR \
   -DCLIENT_ONLY=TRUE \
   -DSWIG_EXECUTABLE=$SWIG_DIR || die "failed to configure UDA"
cmake3 --build build/ || die "failed to build UDA"
cmake3 --build build/ --target install || die "failed to install UDA"




