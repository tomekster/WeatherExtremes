#!/usr/bin/env python

import cdsapi 

c = cdsapi.Client() 

YEARS   = [ "1940", "1941", "1942", "1943", "1944", "1945", "1946", "1947", "1948", "1949",
			"1950", "1951", "1952", "1953", "1954", "1955", "1956", "1957", "1958", "1959",
     		"1960", "1961", "1962", "1963", "1964", "1965", "1966", "1967", "1968", "1969",
			"1970", "1971", "1972", "1973", "1974", "1975", "1976", "1977", "1978", "1979",
			"1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989",  
			"1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999",
			"2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009",
			"2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
			"2020", "2021", "2022" ] 

YEARS   = [ "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
                       "2020", "2021", "2022" ]

MONTHS = [ "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12" ] 

#MONTHS = [ "01" ]
#YEARS  = [ "2023" ]
           
# ----- T2M-MAX -------------------------------------------------------
   
for year in YEARS:           
	for month in MONTHS: 
		filename = "T2M.MEAN-"+year+month+".nc"
		print(filename)
		result = c.service( "tool.toolbox.orchestrator.workflow", 
			params=
				{ 
				"realm": "user-apps", 
				"project": "app-c3s-daily-era5-statistics", 
				"version": "master", 
				"kwargs": 
					{ 
						"dataset": "reanalysis-era5-single-levels", 
						"product_type": "reanalysis", 
						"variable": "2m_temperature", 
						"statistic": "daily_mean", 
						"year": year, 
						"month": month, 
						"time_zone": "UTC+00:00", 
						"frequency": "1-hourly", 
						"grid": "0.25/0.25", 
						"area": {"lat": [-90, 90], "lon": [-180, 180]}, 
					}, 
				"workflow_name": "application" 
				}) 
				
		c.download(result)

		
