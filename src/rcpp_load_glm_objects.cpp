#include "rcpp_load_glm_objects.h"

#include <Rcpp.h>

using namespace Rcpp;

template <> glmFamily* Rcpp::as(SEXP familySexp) {
	List familyList = List(familySexp);
	std::string linkName = as<std::string>(familyList["link"]);
	std::string varianceName = as<std::string>(familyList["variance"]);

	return new glmFamily(linkName, varianceName);
}

template <> glmData* Rcpp::as(SEXP dataSexp) {
	glmData* results;

	List dataList = List(dataSexp);
	List terms = dataList["terms"];
	List xNumericTerms = terms["numeric.terms"];
	int nXNumeric = xNumericTerms.size();

	NumericVector y = as<NumericVector>(dataList["response"]);
	NumericVector xTemp;
	NumericVector weights;

	num_t *yPointer = (num_t*) (&y[0]);
	num_t **xNumericPointer;
	num_t *weightsPointer;

	xNumericPointer = (num_t **) malloc(nXNumeric * sizeof(num_t *));
	for (int i = 0; i < nXNumeric; i++) {
		xTemp = as<NumericVector>(xNumericTerms[i]);
		xNumericPointer[i] = (num_t*) (&xTemp[0]);
	}

	if (dataList.containsElementNamed("weights")) {
		weights = as<NumericVector>(dataList["weights"]);
		weightsPointer = (num_t*) (&weights[0]);
	} else {
		weightsPointer = NULL;
	}

	results = new glmData(y.length(), yPointer, xNumericPointer,
			nXNumeric, weightsPointer);

	free(xNumericPointer);

	return results;
}

template <> glmControl* Rcpp::as(SEXP controlSexp) {
	List controlList = List(controlSexp);

	std::string fitMethod = as<std::string>(controlList["fit.method"]);
	unsigned int maxIterations =
			as<unsigned int>(controlList["max.iterations"]);
	double tolerance = as<double>(controlList["tolerance"]);

	return new glmControl(fitMethod, maxIterations, tolerance);
}

template <> glmObject* Rcpp::as(SEXP objectSexp) {
	List objectList = List(objectSexp);

	glmData *data = as<glmData*>(objectList["data"]);
	glmFamily *family = as<glmFamily*>(objectList["family"]);
	glmControl *control = as<glmControl*>(objectList["control"]);

	return new glmObject(data, family, control);
}

template <> SEXP Rcpp::wrap(const glmVector<num_t> &x) {
	std::vector<num_t> xStd = std::vector<num_t>(x.getHostData(),
			x.getHostData() + x.getLength());

	return wrap(xStd);
}

template <> SEXP Rcpp::wrap(const glmResults &results) {
	List resultsList = List();

	resultsList["beta"] = wrap(*(results.getBeta()));
	resultsList["num.iterations"] = wrap(results.getNumIterations());
	resultsList["converged"] = wrap(results.getConverged());

	return wrap(resultsList);
}

void print_gpu_status() {
	size_t free, total;

	cudaMemGetInfo(&free, &total);

	std::cout << "CUDA status: " << free << " out of " << total
			<< " bytes free." << std::endl;

	return;
}

// [[Rcpp::export]]
SEXP cpp_gpu_glm(SEXP objectSexp) {
	SEXP results= NULL;

	print_gpu_status();

	try {
		glmObject *glmObj = as<glmObject*>(objectSexp);

		glmObj->solve();

		results = wrap(*(glmObj->getResults()));
		delete glmObj;
	} catch (glmCudaException e) {
		Rcout << "CUDA ERROR: " << e.what() << " (Code "
				<< e.getCudaErrorCode() << ")" << std::endl;
	} catch (glmCublasException e) {
		Rcout << "CUBLAS ERROR: " << e.what() << std::endl;
	} catch (glmCusolverException e) {
		Rcout << "CUSOLVER ERROR: " << e.what() << std::endl;
	}

	return results;
}
