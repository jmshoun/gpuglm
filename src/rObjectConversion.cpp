#include "rObjectConversion.h"

#include <Rcpp.h>

using namespace Rcpp;

// R object to glmVector and glmMatrix conversion /////////////////////////////

num_t* rVectorToPointer(SEXP rVector) {
	NumericVector vector = as<NumericVector>(rVector);
	return (num_t*) &(vector[0]);
}

glmVector<num_t>* rToNumVector(SEXP rVector) {
	NumericVector vector = as<NumericVector>(rVector);
	num_t *dataPointer = (num_t*) &(vector[0]);

	return new glmVector<num_t>(dataPointer, vector.length());
}

glmMatrix<num_t>* rToNumMatrix(SEXP rVectorList) {
	List matrixColumns = List(rVectorList);
	NumericVector tempColumn = as<NumericVector>(matrixColumns[0]);
	int nColumns = matrixColumns.size();
	int nRows = tempColumn.length();
	num_t *hostData;

	glmMatrix<num_t> *result = new glmMatrix<num_t>(tempColumn.length(),
			nColumns, false, true, false);
	for (int i = 0; i < nColumns; i++) {
		result->copyColumnFromHost(rVectorToPointer(matrixColumns[i]), i);
	}

	return result;
}

glmMatrix<factor_t>* rToFactorMatrix(SEXP rVectorList) {
	List matrixColumns = List(rVectorList);
	IntegerVector tempColumn = as<IntegerVector>(matrixColumns[0]);
	int nColumns = matrixColumns.size();
	int nRows = tempColumn.length();
	factor_t *tempFactor = (factor_t*) malloc(sizeof(factor_t) * nRows);

	glmMatrix<factor_t> *result = new glmMatrix<factor_t>(nRows, nColumns,
			false, true, false);
	for (int i = 0; i < nColumns; i++) {
		tempColumn = as<IntegerVector>(matrixColumns[i]);
		for (int j = 0; j < nRows; j++) {
			tempFactor[i] = (factor_t) tempColumn[i];
		}
		result->copyColumnFromHost(tempFactor, i);
	}

	free(tempFactor);
	return result;
}

// R object to C++ object conversion //////////////////////////////////////////

template <> glmObject* Rcpp::as(SEXP objectSexp) {
	List objectList = List(objectSexp);

	glmData *data = as<glmData*>(objectList["data"]);
	glmFamily *family = as<glmFamily*>(objectList["family"]);
	glmControl *control = as<glmControl*>(objectList["control"]);

	return new glmObject(data, family, control);
}

template <> glmFamily* Rcpp::as(SEXP familySexp) {
	List familyList = List(familySexp);

	std::string linkName = as<std::string>(familyList["link"]);
	std::string varianceName = as<std::string>(familyList["variance"]);
	bool isCanonical = as<bool>(familyList["is.canonical"]);
	num_t scaleParameter = (num_t) as<double>(familyList["scale.parameter"]);

	return new glmFamily(linkName, varianceName, isCanonical, scaleParameter);
}

template <> glmControl* Rcpp::as(SEXP controlSexp) {
	List controlList = List(controlSexp);

	std::string fitMethod = as<std::string>(controlList["fit.method"]);
	unsigned int maxIterations =
			as<unsigned int>(controlList["max.iterations"]);
	double tolerance = as<double>(controlList["tolerance"]);

	return new glmControl(fitMethod, maxIterations, tolerance);
}

template <> glmData* Rcpp::as(SEXP dataSexp) {
	glmVector<num_t> *y;
	glmMatrix<num_t> *xNumeric = NULL;
	glmVector<num_t> *weights = NULL;

	List dataList = List(dataSexp);
	List terms = dataList["terms"];

	y = rToNumVector(dataList["response"]);
	y->copyHostToDevice();

	xNumeric = rToNumMatrix(terms["numeric.terms"]);
	if (dataList.containsElementNamed("weights")) {
		weights = rToNumVector(dataList["weights"]);
	}

	return new glmData(y, xNumeric, NULL, weights);
}

// C++ object to R object conversion //////////////////////////////////////////

template <> SEXP Rcpp::wrap(const glmVector<num_t> &x) {
	std::vector<num_t> xStd = std::vector<num_t>(x.getHostData(),
			x.getHostData() + x.getLength());

	return wrap(xStd);
}

template <> SEXP Rcpp::wrap(const glmResults &results) {
	List resultsList = List();

	results.getBeta()->copyDeviceToHost();

	resultsList["beta"] = wrap(*(results.getBeta()));
	resultsList["num.iterations"] = wrap(results.getNumIterations());
	resultsList["converged"] = wrap(results.getConverged());
	resultsList["status"] = "SUCCESS";

	return wrap(resultsList);
}
