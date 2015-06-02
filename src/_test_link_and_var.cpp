#include <Rcpp.h>

#include <vector>
#include <string>

#include "linkFunctions.h"
#include "varianceFunctions.h"

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector test_link(NumericVector x, std::string linkType, double k = 0) {
	glmVector<num_t> data((num_t *) &x[0], x.size(), true);
  linkFunction link = getLinkFunction(linkType);

	data.copyHostToDevice();
	link(&data, k);
	data.copyDeviceToHost();
  
	std::vector<num_t> result(data.getHostData(), data.getHostData() + data.getLength());
	return wrap(result);
}

// [[Rcpp::export]]
NumericVector test_inv_link(NumericVector x, std::string linkType, double k = 0) {
  glmVector<num_t> data((num_t *) &x[0], x.size(), true);
  linkFunction link = getInvLinkFunction(linkType);

	data.copyHostToDevice();
	link(&data, k);
	data.copyDeviceToHost();
  
	std::vector<num_t> result(data.getHostData(), data.getHostData() + data.getLength());
	return wrap(result);
}

// [[Rcpp::export(test_variance)]]
NumericVector test_variance(NumericVector x, std::string varianceType, double k = 0) {
  glmVector<num_t> data((num_t *) &x[0], x.size(), true);
  varianceFunction variance = getVarianceFunction(varianceType);

  data.copyHostToDevice();
	variance(&data, k);
	data.copyDeviceToHost();
  
	std::vector<num_t> result(data.getHostData(), data.getHostData() + data.getLength());
	return wrap(result);
}
