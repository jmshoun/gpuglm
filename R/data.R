.create_glm_data <- function(formula, data, weights=NULL) {
  .main <- function() {
    glm.data <- list()
    model.terms <- terms(formula, data=data)
    factor.matrix <- attr(model.terms, 'factors')
    
    glm.data$response <- .get_response(factor.matrix)
    glm.data$terms <- .get_terms(factor.matrix)
    if (!is.null(weights)) {
      glm.data$weights <- .get_weights() %>%
        .validate_weights(length(glm.data$terms[[1]]))
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
    lapply(term.names, function(term.name) {
      .extract_factor(term.name)
    }) %>%
      magrittr::set_names(term.names)
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

.validate_weights <- function(weights, data.length) {
  if (length(weights) != data.length) {
    stop('Weights must be the same length as data')
  }
  
  min.weight <- min(weights)
  
  if (min.weight < 0) {
    warn('Negative weights found in data. Setting negative weights to zero...')
    weights <- pmax(weights, 0)
  } else if (min.weight == 0) {
    warn('Zero weights found in data. Zero weight observations will be ignored...')
  }
  
  weights
}