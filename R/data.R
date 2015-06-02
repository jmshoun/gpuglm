.create_glm_data <- function(formula, data, weights=NULL) {
  .main <- function() {
    glm.data <- list()
    model.terms <- terms(formula, data=data)
    factor.matrix <- attr(model.terms, 'factors')
    
    glm.data$response <- .get_response(factor.matrix)
    glm.data$terms <- .get_terms(factor.matrix)
    if (!is.null(weights)) {
      glm.data$weights <- .get_weights()
    }
    
    glm.data
  }
  
  .check_for_interactions <- function(model.terms) {
    max.order <- max(attr(model.terms, 'order'))
    if (max.order > 1) {
      stop('Interactions are not supported yet.')
    }
  }
  
  .get_response <- function(factor.matrix) {
    response.name <- rownames(factor.matrix)[1]
    .extract_factor(response.name)
  }
  
  .get_terms <- function(factor.matrix) {
    term.names <- rownames(factor.matrix)[-1]
    is.term.numeric <- sapply(term.names, function(term.name) {
      parse(text=term.name) %>%
        eval(envir=data) %>%
        is.numeric()
    })
    
    if (!all(is.term.numeric)) {
      stop('Factor terms are not supported yet.')
    }
    
    list(numeric.terms=.get_numeric_terms(term.names))
  }
  
  .get_numeric_terms <- function(term.names) {
    sapply(term.names, function(term.name) {
      .extract_factor(term.name)
    })
  }
  
  .get_weights <- function() {
    tryCatch({
      substitute(weights) %>%
        eval(envir=data)
    }, error=function(e) {
      stop('Specified weights not found')
    })
  }
  
  .extract_factor <- function(factor.name) {
    if (factor.name %in% names(data)) {
      data[[factor.name]]
    } else {
      parse(text=factor.name) %>%
        eval(envir=data)
    }
  }
  
  .main()
}