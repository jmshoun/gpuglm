.create_glm_data <- function(formula, data, weights.call=NULL) {
  .main <- function() {
    glm.data <- list()
    
    model.terms <- terms(formula, data=data) %T>%
      .check_for_interactions()
    factor.matrix <- attr(model.terms, 'factors')
    
    glm.data$response <- .get_response(factor.matrix)
    glm.data$terms <- .get_terms(factor.matrix)
    
    if (!is.null(weights.call)) {
      glm.data$weights <- .get_weights() %>%
        .validate_weights(length(glm.data$terms[[1]][[1]]))
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
    unsorted.terms <- sapply(term.names, .extract_factor, simplify=FALSE)
    
    is.numeric.term <- sapply(unsorted.terms, is.numeric)
    is.factor.term <- sapply(unsorted.terms, is.factor)
    if (!all(is.numeric.term | is.factor.term)) {
      stop('Only numeric and factor terms are currently supported.')
    }
    
    terms <- list()
    if (any(is.numeric.term)) {
      terms$numeric.terms <- unsorted.terms[is.numeric.term]
    }
    if (any(is.factor.term)) {
      terms$factor.terms <- unsorted.terms[is.factor.term]
      terms$factor.offsets <- .set_factor_offsets(terms)
    }
    
    terms
  }
  
  .get_weights <- function() {
    tryCatch({
      eval(weights.call, data)
    }, error=function(e) {
      stop('Specified weights not found')
    })
  }
  
  .extract_factor <- function(factor.name) {
    parse(text=factor.name) %>%
      eval(envir=data)
  }
  
  .set_factor_offsets <- function(terms) {
    initial.offset <- length(terms$numeric.terms)
    factor.level.counts <- sapply(terms$factor.terms, function(factor.term) {
      levels(factor.term) %>%
        length()
    })
    
    initial.offset + c(0, head(factor.level.counts - 1, -1))
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