.build_starting_beta <- function(terms, prior.fit) {
  prior.beta <- prior.fit$beta
  
  starting.intercept <- 0
  numeric.names <- names(terms$numeric.terms)
  starting.numerics <- rep(0, length(terms$numeric.terms)) %>%
    magrittr::set_names(numeric.names)
  
  factor.names <- names(terms$factor.terms)
  starting.factors <- sapply(terms$factor.terms, function(factor.term) {
    rep(0, nlevels(factor.term) - 1)
  }, simplify=FALSE)
  
  if (!is.null(prior.beta$intercept)) {
    starting.intercept <- prior.beta$intercept
  }
  
  if (!is.null(prior.beta$numeric)) {
    starting.numerics[numeric.names] <- prior.beta$numeric[numeric.names] %>%
      .convert_na_to_zero()
  }
  
  c(starting.numerics, unlist(starting.factors), starting.intercept)
}