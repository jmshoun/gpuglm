install: gpuglm.so

linkFunctions.o: linkFunctions.cu
	/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -c linkFunctions.cu -o linkFunctions.o -lcudart

varianceFunctions.o: varianceFunctions.cu
	/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -c varianceFunctions.cu -o varianceFunctions.o -lcudart

glmArray.o: glmArray.cu
	/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -c glmArray.cu -o glmArray.o -lcudart -lcublas

glmObject.o: glmObject.cu
	/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -c glmObject.cu -o glmObject.o -lcudart -lcublas -lcusolver

RcppExports.o: RcppExports.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c RcppExports.cpp -o RcppExports.o

glmData.o: glmData.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c glmData.cpp -o glmData.o -lcudart -I/usr/local/cuda/include

_test_link_and_var.o: _test_link_and_var.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c _test_link_and_var.cpp -o _test_link_and_var.o -lcudart -I/usr/local/cuda/include

rObjectConversion.o: rObjectConversion.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c rObjectConversion.cpp -o rObjectConversion.o -lcudart -I/usr/local/cuda/include

rcpp_glm_interface.o: rcpp_glm_interface.cpp
	g++ -I/home/mark/R/include -DNDEBUG  -I/usr/local/include -I"/home/mark/R/library/Rcpp/include" -fpic -g -O2 -c rcpp_glm_interface.cpp -o rcpp_glm_interface.o -lcudart -I/usr/local/cuda/include

gpuglm.so: RcppExports.o _test_link_and_var.o linkFunctions.o varianceFunctions.o rcpp_glm_interface.o glmData.o glmObject.o glmArray.o rObjectConversion.o
	g++ -shared -L/home/mark/R/lib -L/usr/local/lib -o gpuglm.so RcppExports.o _test_link_and_var.o linkFunctions.o varianceFunctions.o glmData.o glmObject.o glmArray.o rcpp_glm_interface.o rObjectConversion.o -L/home/mark/R/lib -L/usr/local/cuda/lib64 -lR -lcudart -lcublas -lcusolver

clean:
	rm *.o *.so

