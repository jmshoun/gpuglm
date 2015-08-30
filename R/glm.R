##' GPU-Acclerated GLM Fitting
##' 
##' \code{gpuglm} is a function that is designed to be more or less analogous to glm(). The class
##' of the object that is returned is significantly different from \code{glm} (for some very good
##' reasons), but most \code{glm} methods have \code{gpuglm} counterparts with very similar syntax.
##' 
##' @param formula A formula that specifies the model structure; entirely analogous to the
##' corresponding argument in \code{glm}.
##' @param family An object of class \code{gpuglm_family}.
##' @param data A \code{data.frame} to fit the model with. Expressions in \code{formula} are first
##' evaluated in the context of \code{data}, and if no match is found, then in the function
##' environment.
##' @param weights A vector with weights for each observation. Must be the same as each of the
##' terms in \code{data}. If \code{NULL}, all observations are assumed to have equal weight.
##' @param control An object of class \code{gpuglm_control} that controls the fitting algorithm and
##' convergence criteria.
##' @param prior.fit An object returned from a prior call to \code{gpuglm}. The parameters
##' associated with \code{prior.fit} are used to set the initial parameter estimates in the current
##' model, which can significantly speed up fit times.
##' 
##' @return An object of class \code{gpuglm}.

gpuglm <- function(formula, family=gpuglm_family(), data, weights, 
                    control=gpuglm_control(), prior.fit=NULL) {
  .main <- function() {
    data <- .create_glm_data(formula, data, weights.call)
    if (is.character(family)) {
      family <- gpuglm_family(family)
    }
    
    glm.object <- structure(list(data=data,
                                 family=family,
                                 control=control),
                            class='gpuglm_specification')
    glm.object$starting.beta <- .build_starting_beta(glm.object$data$terms, prior.fit)
    
    results <- cpp_gpu_glm(glm.object)
    
    if (results$status == 'FAILED') {
      stop("Execution on GPU failed")
    } else {
      if (!(results$converged)) {
        warning('Model failed to converge')
      }
      .format_results(results,glm.object)
    }
  }
  
  .format_results <- function(results, initial.object) {
    results$status <- NULL
      
    raw.beta <- results$beta
    intercept <- raw.beta[length(raw.beta)] %>%
      magrittr::set_names('(Intercept)')
    numeric.betas <- raw.beta[1:length(initial.object$data$terms$numeric.terms)] %>%
      magrittr::set_names(names(initial.object$data$terms$numeric.terms))
    factor.betas <- .format_factor_betas(raw.beta, initial.object)
    
    results$beta <- list(intercept=intercept,
                         numeric=numeric.betas,
                         factor=factor.betas)
    results$call <- raw.call
    attr(results, 'class') <- 'gpuglm'
    
    results
  }
  
  .format_factor_betas <- function(raw.beta, initial.object) {
    factor.betas <- NULL
    if (!is.null(initial.object$data$terms$factor.offsets)) {
      factor.terms <- initial.object$data$terms$factor.terms
      factor.offsets <- initial.object$data$terms$factor.offsets
      factor.lengths <- initial.object$data$terms$factor.lengths
      factor.betas <- list()
      
      for (i in 1:length(factor.terms)) {
        factor.indices <- factor.offsets[i]:(factor.offsets[i] + factor.lengths[i] - 1) + 3
        factor.betas[[names(factor.terms)[i]]] <- raw.beta[factor.indices]
        names(factor.betas[[names(factor.terms)[i]]]) <- levels(factor.terms[[i]])[-1]
      }
    }
    
    factor.betas
  }
  
  raw.call <- match.call()
  if (missing(weights)) {
    weights.call <- NULL
  } else {
    weights.call <- substitute(weights)
  }
  
  .main()
}
