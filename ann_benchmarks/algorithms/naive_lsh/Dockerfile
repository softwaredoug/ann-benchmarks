# The base image managed by this project
FROM ann-benchmarks

# Here I'm manually installing this module, not on pypi
# This is just a numpy C extension of rand projection based
# locality-sensitive hashing (LSH) which uses hamming distance
# to approximate cosine distance
RUN git clone https://github.com/softwaredoug/ufuncs.git
RUN cd ufuncs; python3 setup.py  build 
RUN cd ufuncs; python3 setup.py  install

# python -c just runs a command
# in this case, this is just a check we can import the module
RUN python3 -c "import similarities"               
