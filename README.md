# WeatherExtremes

# Available Scripts

./src/preprocess - preprocess weatherbench2_original data to daily values format expected by WeatherExtremes pipeline
./src/main.py - basic experiments: configure the dataset, aggregations, reference period, analysis period, percentiles, etc. and produce the percentiles file and the exceedances file
./app.py - visualisation tool for exploring how percentiles change in different locations on earth

./serc/vis/perc_vis.py - takes percentiles file path, timestep, and outpath as input and visualises the percentiles  


# Installation:
External dependencies for Cartopy 

On MacOS:
- brew install cdo
- make sure you have installed Xcode and accepted the license by running: sudo xcodebuild -license
- you might have to install additional external dependencies:
     brew install geos proj gdal

On Linux:
- apt-get install cdo

From the root directory of the project run the following commands:
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
