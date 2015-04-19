library(gpuglm)

context("Link and Inverse Link Functions")

test_link_function <- function(exact.link.values, exact.response.values, link.type, k=0) {
  expect_equal(test_link(exact.response.values, link.type, k),
               exact.link.values, tolerance=1e-6)
  expect_equal(test_inv_link(exact.link.values, link.type, k), 
               exact.response.values, tolerance=1e-6)
  
  mutated.values <- test_link(test_inv_link(exact.link.values, link.type, k), 
                              link.type, k)
  expect_equal(exact.link.values, mutated.values, tolerance=1e-6)
}

test_that("Identity link functions correctly", {
  ## Exact value checking
  test.values <- c(-1e6, -10:10, 1e6)
  test_link_function(test.values, test.values, 'identity')
})

test_that("Reciprocal link functions correctly", {
  ## Exact value checking
  exact.link.values <- c(.02, .1, .5, 1, 5, 10, 50)
  exact.response.values <- 1 / exact.link.values
  test_link_function(exact.link.values, exact.response.values, 'reciprocal')
})

test_that("Squared reciprocal link functions correctly", {
  ## Exact value checking
  exact.link.values <- c(.02, .1, .5, 1, 5, 10, 50)
  exact.response.values <- 1 / exact.link.values ** .5
  test_link_function(exact.link.values, exact.response.values, 'squared reciprocal')
})

test_that("Log link functions correctly", {
  ## Exact value checking
  exact.response.values <- c(1:30)
  exact.link.values <- log(exact.response.values)
  test_link_function(exact.link.values, exact.response.values, 'log')
  
  ## Range checking
  expect_equal(test_inv_link(-1e6, 'log'), 0)
})

test_that("Logit link functions correctly", {
  ## Exact value checking
  exact.response.values <- seq(.05, .95, .05)
  exact.link.values <- log(exact.response.values / (1 - exact.response.values))
  test_link_function(exact.link.values, exact.response.values, 'logit')
  
  ## Range checking
  expect_equal(test_inv_link(c(-1000, 1000), 'logit'), c(0, 1))
})

test_that("Negative binomial link functions correctly", {
  ## Exact value checking
  k <- 5
  exact.response.values <- 1:30
  exact.link.values <- log(exact.response.values / (exact.response.values + k))
  test_link_function(exact.link.values, exact.response.values, 'negative binomial', k)
})