# PyQuantizerEncoder
A simple quantizers encoders(CIQA, CIVQ, CIMap, Dithering quantizer) project to the discipline Signal Compression 2/2020

## Virtual env

- To run this project in a virtual enviroment with custom python libs, install python 
virtualenv and execute:

$ virtualenv venv

and to activate the virtual env execute:

$ source /venv/bin/activate

## Running files

### CIQA 

Run:

$ python CIQA.py N M Image\ Database/image_name.bmp       

Where N and M is a integer.

### CIQA 

Run:

$ python CIQA.py N M Image\ Database/image_name.bmp       

Where N and M is a integer.

### CIVQ

Run:

$ python CIVQ.py L M Image\ Database/image_name.bmp       

Where L and M is a integer.

### CIMap

Run:

$ python3 CIMap.py M Image\ Database/image_name.bmp   

Where M is a integer.

### FloydSteinbergGray

Run:

$ python FloydSteinbergGray.py 8 Image\ Database/image_name.bmp

Where M is a integer.

### FloydSteinbergColor

Run:

$ python FloydSteinbergColor.py 8 Image\ Database/image_name.bmp

Where M is a integer.
