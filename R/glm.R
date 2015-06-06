##' GPU-Acclerated GLM Fitting
##' 
##' \code{gpuglm} is a function that is designed to be more or less analogous to glm(). The class
##' of the object that is returned is significantly different from \code{glm} (for some very good
##' reasons), but most \code{glm} methods have \code{gpuglm} counterparts with very similar syntax.
##' 
##' @param formula A formula that specifies the model structure; entirely analogous to the
##' corresponding argument in \code{glm}.
##' @param data A \code{data.frame} to fit the model with. Expressions in \code{formula} are first
##' evaluated in the context of \code{data}, and if no match is found, then in the function
##' environment.
##' @param family An object of class \code{gpuglm_family}.
##' @param weights A vector with weights for each observation. Must be the same as each of the
##' terms in \code{data}. If \code{NULL}, all observations are assumed to have equal weight.
##' @param control An object of class \code{gpuglm_control} that controls the fitting algorithm and
##' convergence criteria.
##' 
##' @return An object of class \code{gpuglm}.

gpuglm <- function(formula, data, family=gpuglm_family(), weights=NULL, 
                    control=gpuglm_control()) {
  
  .main <- function() {
    data <- .create_glm_data(formula, data, weights)
    glm.object <- structure(list(data=data,
                                 family=family,
                                 control=control),
                            class='gpuglm_specification')
    
    results <- cpp_gpu_glm(glm.object)
    if (!(results$converged)) {
      warning('Model failed to converge')
    }
    .format_results(results, data=data)
  }
  
  .format_results <- function(data, results) {
    old.beta <- results$beta
    intercept <- old.beta[length(old.beta)] %>%
      magrittr::set_names('(Intercept)')
    numeric.betas <- old.beta[1:(length(old.beta) - 1)] %>%
      magrittr::set_names(names(data$terms$numeric.terms))
    results$beta <- list(intercept=intercept,
                         numeric=numeric.betas)
    attr(results, 'class') <- 'gpuglm'
    
    results
  }
  
  .main()
}