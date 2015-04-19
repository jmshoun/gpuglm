library(gpuglm)

context("Variance Functions")

test_that("Identity variance functions correctly", {
  test.values <- c(1:10, 1e6)
  expect_equal(test_variance(test.values, 'identity'), test.values)
})

test_that("Constant variance functions correctly", {
  test.values <- c(-1e6, -10:10, 1e6)
  expect_equal(test_variance(test.values, 'constant'), rep(1, length(test.values)))
})

test_that("Squared variance functions correctly", {
  test.values <- c(1:30)
  expected.values <- test.values ** 2
  expect_equal(test_variance(test.values, 'squared'), expected.values)
})

test_that("Cubed variance functions correctly", {
  test.values <- c(1:30)
  expected.values <- test.values ** 3
  expect_equal(test_variance(test.values, 'cubed'), expected.values)
})

test_that("Binomial variance functions correctly", {
  test.values <- seq(.02, .98, .02)
  expected.values <- test.values * (1 - test.values)
  expect_equal(test_variance(test.values, 'binomial'), expected.values)
})

test_that("Negative binomial variance functions correctly", {
  k <- 4.7
  test.values <- 1:25
  expected.values <- test.values + test.values ** 2 / k
  expect_equal(test_variance(test.values, 'negative binomial', k), expected.values)
})