gpu_glm <- function(formula, data, weights=NULL, family=NULL,
                    control=gpuglm_control()) {
  
  .main <- function() {
    data <- .create_glm_data(formula, data, weights)
    structure(list(data=data,
                   family=family,
                   control=control),
              class='gpu_glm')
  }
  
  .main()
}