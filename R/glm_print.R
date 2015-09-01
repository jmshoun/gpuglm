coef.gpuglm <- function(object, ...) {
  unlist(object$beta)
}

print.gpuglm <- function(x, ...) {
  cat('\nCall:\n')
  print(x$call)
  
  cat('\nNumeric Coefficients:\n')
  print(c(x$beta$intercept, x$beta$numeric))
  
  cat('\nFactor Coefficeints:\n')
  for (i in 1:length(x$beta$factor)) {
    cat(names(x$beta$factor)[i], '\n', sep='')
    print(x$beta$factor[[i]])
  }
  
  cat('\nResidual Deviance: ', -2 * x$log.likelihood, 
      ' AIC: ', -2 * (x$log.likelihood - length(coef(x))), 
      '\n', sep='')
  
  invisible(x)
}
