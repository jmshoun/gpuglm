coef.gpuglm <- function(object, ...) {
  unlist(object$beta)
}

print.gpuglm <- function(x, ...) {
  cat('\nCall:\n')
  print(x$call)
  
  cat('\nNumeric Coefficients:\n')
  print(c(x$beta$intercept, x$beta$numeric))
  
  cat('\nFactor Coefficeints:\n')
  for (i in length(x$beta$factor)) {
    cat(names(x$beta$factor)[i], '\n', sep='')
    print(x$beta$factor[[i]])
  }
  cat('\n')
  
  invisible(x)
}
