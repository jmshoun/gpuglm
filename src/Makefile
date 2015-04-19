install: gpuglm.so

linkFunctions.o: linkFunctions.cu
	/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -c linkFunctions.cu -o linkFunctions.o -lcudart

RcppExports.o: RcppExports.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c RcppExports.cpp -o RcppExports.o

_test_links.o: _test_links.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c _test_links.cpp -o _test_links.o -lcudart -I/usr/local/cuda/include

gpuglm.so: RcppExports.o _test_links.o linkFunctions.o
	g++ -shared -L/home/mark/R/lib -L/usr/local/lib -o gpuglm.so RcppExports.o _test_links.o linkFunctions.o -L/home/mark/R/lib -L/usr/local/cuda/lib64 -lR -lcudart

clean:
	rm *.o *.so
