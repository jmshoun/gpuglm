#ifndef ROBJECTCONVERSION_H_
#define ROBJECTCONVERSION_H_

#include <RcppCommon.h>
#include "glmObjectNR.h"

namespace Rcpp {
	template <> glmObjectNR* as(SEXP objectSexp);
	template <> glmFamily* as(SEXP familySexp);
	template <> glmData* as(SEXP dataSexp);
	template <> glmControl* as(SEXP controlSexp);

	template <> SEXP wrap(const glmResults &results);
	template <> SEXP wrap(const glmVector<num_t> &x);
}

#endif /* ROBJECTCONVERSION_H_ */
