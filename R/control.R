##' GPU-GLM Control Object
##' 
##' \code{gpuglm_family} is the R-side object that contains details about the method to be used in
##' fitting GLM parameters, as well as control parameters associated with the fitting algorithm.
##' 
##' @param fit.method A string with the name of the fit method. Currently, \code{IRLS} is the only 
##' method that is implemented, but \code{BFGS} and \code{L-BFGS} will be implemented eventually.
##' @param max.iterations The maximum number of parameter update iterations for the algorithm. In
##' general, \code{IRLS} will converge in fewer iterations than the other methods.
##' @param tolerance The convergence threshold for the algorithm -- defined as the \eqn{L_{2}} norm 
##' of the change in the parameter vector between consecutive iterations of the algorithm.
##' @return A object of class \code{gpuglm_control}.
gpuglm_control <- function(fit.method='IRLS', max.iterations, tolerance) {
  max.iterations.defaults <- c(IRLS=10,
                               BFGS=30,
                               `L-BFGS`=40)
  tolerance.defaults <- c(IRLS=1e-6,
                          BFGS=1e-6,
                          `L-BFGS`=1e-6)
  
  if (!(fit.method) %in% c('IRLS', 'BFGS', 'L-BFGS')) {
    warning('Unknown fit method. See documentation for acceptable values.\n',
            'Defaulting to IRLS.')
  }
  
  if (missing(max.iterations)) {
    max.iterations <- max.iterations.defaults[fit.method]
  }
  
  if (missing(tolerance)) {
    tolerance <- tolerance.defaults[fit.method]
  } else {
    if (tolerance <= 0) {
      warning('Tolerance must be strictly positive.\n',
              'Defaulting to the default value for fit.method ', fit.method)
      tolerance <- tolerance.defaults[fit.method]
    }
  }
  
  structure(list(fit.method=fit.method,
                 max.iterations=max.iterations,
                 tolerance=tolerance),
            class='gpuglm_control')
}