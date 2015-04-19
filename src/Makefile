install: gpuglm.so

linkFunctions.o: linkFunctions.cu
	/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -c linkFunctions.cu -o linkFunctions.o -lcudart

varianceFunctions.o: varianceFunctions.cu
	/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -c varianceFunctions.cu -o varianceFunctions.o -lcudart

RcppExports.o: RcppExports.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c RcppExports.cpp -o RcppExports.o

_test_link_and_var.o: _test_link_and_var.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c _test_link_and_var.cpp -o _test_link_and_var.o -lcudart -I/usr/local/cuda/include

gpuglm.so: RcppExports.o _test_link_and_var.o linkFunctions.o varianceFunctions.o
	g++ -shared -L/home/mark/R/lib -L/usr/local/lib -o gpuglm.so RcppExports.o _test_link_and_var.o linkFunctions.o varianceFunctions.o -L/home/mark/R/lib -L/usr/local/cuda/lib64 -lR -lcudart

clean:
	rm *.o *.so

