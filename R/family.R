.CANONICAL_LINKS <- data.frame(family=c('gaussian', 'poisson', 'binomial', 'Gamma', 
                                             'inverse.gaussian', 'negative.binomial'),
                               link=c('identity', 'log', 'logit', 'reciprocal', 
                                      'squared reciprocal', 'negative binomial'),
                               variance=c('constant', 'identity', 'binomial',  'squared', 'cubed',
                                          'negative binomial'))

##' GPU-GLM Family Object
##' 
##' \code{gpuglm_family} is the R-side object that contains details about the family to be use in a
##' generalized linear model.
##' 
##' The family for a GLM is defined as the combination of the link and variance functions 
##' associated with the model. (There is also an inverse link function, but it is deteremined by
##' the link function.)
##' 
##' Please note: passing a family name to this function will automatically override any supplied
##' \code{link} and \code{variance} values.
##' 
##' @param family A string with the name of the family. All of the standard \code{glm} families
##' are defined except for the \code{quasi} families. The \code{negative.binomial} family is also
##' defined.
##' @param link A case-insensitive string with the name of the link function. The following link
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
gpuglm_family <- function(family, link='identity', variance='constant') {
  if (!missing(family)) {
    .validate_family_parameter(family, 'family')
    
    link <- .CANONICAL_LINKS[.CANONICAL_LINKS$family == family, 'link']
    variance <- .CANONICAL_LINKS[.CANONICAL_LINKS$family == family, 'variance']
  }
  
  .validate_family_parameter(link, 'link')
  .validate_family_parameter(variance, 'variance')
  
  structure(list(link=tolower(link),
                 variance=tolower(variance),
                 canonical=.determine_canonicity(link, variance),
                 class='gpuglm_family'))
}

.validate_family_parameter <- function(parameter, parameter.name) {
  if (!(parameter %in% .CANONICAL_LINKS[[parameter.name]])) {
    stop('Unrecognized ', parameter.name, ' name')
  }
}

.determine_canonicity <- function(link, variance) {
  which(.CANONICAL_LINKS$link == link) == which(.CANONICAL_LINKS$variance == variance)
}