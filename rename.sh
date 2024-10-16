#!/bin/csh

set mode = t2mean 

# T2MEAN 
if ( "${mode}" == "t2mean" ) then
   foreach file ( `ls -1tr *.nc` )
      set datestr = `ncdump -h ${file} | grep 'time:units' | awk '{print $5}' | cut -c 1-7`
      \mv -f ${file} t2mean/T2MEAN-${datestr}
    end
endif


# T2MAX
if ( "${mode}" == "t2max" ) then
  foreach file ( `ls -1tr *.nc` )
     set datestr = `ncdump -h ${file} | grep 'time:units' | awk '{print $5}' | cut -c 1-7`
     \mv -f ${file} t2max/T2MAX-${datestr}
   end
endif

# PRECIP
if ( "${mode}" == "precip" ) then
  foreach file ( `ls -1tr *.nc` )
     set datestr = `ncdump -h ${file} | grep 'time:units' | awk '{print $5}' | cut -c 1-7`
     set yyyy    = `echo ${datestr} | cut -c 1-4`
     set mm      = `echo ${datestr} | cut -c 6-7`
     set mm      = `echo ${mm} + 1 | bc`
     set mm      = `printf '%02d' ${mm}`
     if ( "${mm}" == "13" ) then
        set mm   = 01
        set yyyy = `echo ${yyyy} + 1 | bc`
     endif
     \cp -f ${file} precip/PRECIP-${yyyy}-${mm}
   end
endif

exit 0
