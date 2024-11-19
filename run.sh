#!/bin/csh

#\rm -f DATA-*
\rm -f PERC
\rm -f CLIM

matlab -nodesktop < diagnostic.m

exit 0
