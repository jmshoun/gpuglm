\name{gpuglm-package}
\alias{gpuglm-package}
\alias{gpuglm}
\docType{package}
\title{
GPU-Accelerated Generalized Linear Models
}
\description{
}
\details{
\tabular{ll}{
Package: \tab gpuglm\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2015-04-15\cr
License: \tab GPL (>= 2)\cr
}}
\author{
J. Mark Shoun
Maintainer: J. Mark Shoun <mark.shoun@gmail.com>
}
\references{}
\keyword{ package }
\examples{
data(mtcars)
gpuglm(am ~ cyl + hp, data=mtcars, family='binomial')
}
