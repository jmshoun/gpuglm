##' GPU-GLM Family Object
##' 
##' \code{gpuglm_family} is the R-side object that contains details about the family to be use in a
##' generalized linear model.
##' 
##' The family for a GLM is defined as the combination of the link and variance functions 
##' assocaiated with the model. (There is also an inverse link function, but it is deteremined by
##' the link function.)
##' 
##' Please note: There is no validation of the link and variance function strings supplied by the
##' user. If the link or variance function strings are undefined in the C++/CUDA code, the (silent)
##' fallback link and variance functions are both \code{'identity'}.
##' 
##' @param link A case-insentive string with the name of the link function. The following link
##' functions are currently defined:
##' \itemize{
##'   \item \code{identity}
##'   \item \code{log}
##'   \item \code{logit}
##'   \item \code{reciprocal}
##'   \item \code{squared reciprocal}
##'   \item \code{negative binomial}
##' }
##' @param variance A case-insensitive string with the name of the variance function. The following
##' variance functions are currently defined:
##' \itemize{
##'   \item \code{constant}
##'   \item \code{identity}
##'   \item \code{squared}
##'   \item \code{cubed}
##'   \item \code{binomial}
##'   \item \code{negative binomial}
##' }
##' @return An object of class \code{gpuglm_family}.
gpuglm_family <- function(link='identity', variance='constant') {
  structure(list(link=tolower(link),
                 variance=tolower(variance),
                 class='gpuglm_family'))
}