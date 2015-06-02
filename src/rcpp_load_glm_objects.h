#include <RcppCommon.h>

#include "glmObject.h"

namespace Rcpp {
	template <> glmObject* as(SEXP objectSexp);
	template <> glmFamily* as(SEXP familySexp);
	template <> glmData* as(SEXP dataSexp);
	template <> glmControl* as(SEXP controlSexp);

	template <> SEXP wrap(const glmResults &results);
	template <> SEXP wrap(const glmVector<num_t> &x);
}
