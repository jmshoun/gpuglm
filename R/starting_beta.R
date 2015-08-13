.build_starting_beta <- function(terms, prior.fit) {
  prior.beta <- prior.fit$beta
  
  starting.intercept <- 0
  numeric.names <- names(terms$numeric.terms)
  starting.numerics <- rep(0, length(terms$numeric.terms)) %>%
    set_names(numeric.names)
  
  if (!is.null(prior.beta$intercept)) {
    starting.intercept <- prior.beta$intercept
  }
  
  if (!is.null(prior.beta$numeric)) {
    starting.numerics[numeric.names] <- prior.beta$numeric[numeric.names] %>%
      .convert_na_to_zero()
  }
  
  c(starting.numerics, starting.intercept)
}