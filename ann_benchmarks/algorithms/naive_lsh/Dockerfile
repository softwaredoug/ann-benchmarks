FROM ann-benchmarks

RUN git clone https://github.com/softwaredoug/ufuncs.git
RUN cd ufuncs; python3 setup.py  build 
RUN cd ufuncs; python3 setup.py  install
RUN pwd

# Is this just a check?
RUN python3 -c "import similarities"               
