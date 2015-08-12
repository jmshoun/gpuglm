#include "rcpp_glm_interface.h"

#include <Rcpp.h>

#include "rObjectConversion.h"

using namespace Rcpp;

// Debugging verbose status functions /////////////////////////////////////////

void print_gpu_status() {
	size_t free, total;

	cudaMemGetInfo(&free, &total);

	std::cout << "CUDA status: " << free << " out of " << total
			<< " bytes free." << std::endl;

	return;
}

// Primary interface functions ////////////////////////////////////////////////

// [[Rcpp::export]]
SEXP cpp_gpu_glm(SEXP objectSexp) {
	SEXP results = NULL;
	List fallbackResults;
	fallbackResults["status"] = "FAILED";

	print_gpu_status();

	try {
		glmObject *glmObj = as<glmObject*>(objectSexp);
		glmObj->solve();
		results = wrap(*(glmObj->getResults()));

		delete glmObj;
	} catch (glmCudaException e) {
		Rcout << "CUDA ERROR: " << e.what() << " (Code "
				<< e.getCudaErrorCode() << ")" << std::endl;
		Rcout << "Error on line " << e.getLineNumber() << " of " <<
				e.getFileName() << std::endl;
	} catch (glmCublasException e) {
		Rcout << "CUBLAS ERROR: " << e.what() << std::endl;
		Rcout << "Error on line " << e.getLineNumber() << " of " <<
						e.getFileName() << std::endl;
	} catch (glmCusolverException e) {
		Rcout << "CUSOLVER ERROR: " << e.what() << std::endl;
		Rcout << "Error on line " << e.getLineNumber() << " of " <<
						e.getFileName() << std::endl;
	}

	if (results == NULL) {
		results = wrap(fallbackResults);
	}

	return results;
}
