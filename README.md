# WeatherExtremes


# Installation:
External dependencies:
Install CDO: brew install cdo

External dependencies for Cartopy 

On MacOS
- make sure you have installed Xcode and accepted the license by running: sudo xcodebuild -license
- you might have to install additional external dependencies for dependencies brew install geos proj gdal

On Linux:
- TODO

From the root directory of the project run the following commands:
python -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
