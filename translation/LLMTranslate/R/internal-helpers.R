# R/internal-helpers.R
# Internal helpers and data objects for LLMTranslate
# These are used by the shiny app in inst/app/app.R and unit tests.
# They are intentionally not exported.

# -------------------------------------------------------------------
# Data objects
# -------------------------------------------------------------------

#' Model specification table
#'
#' Columns:
#' - name: canonical model id
#' - provider: "openai", "gemini", or "claude"
#' - type: "chat" or "reasoning"
#' - supports_temp: whether temperature should be sent
#'
#' @keywords internal
#' @noRd
MODEL_SPEC <- data.frame(
  name = c(
    # OpenAI chat
    "gpt-4o-mini","gpt-4o","gpt-4.1-mini","gpt-4.1",
    # OpenAI reasoning (o-series)
    "o3-mini","o1-mini","o1",
    # OpenAI reasoning (GPT-5 family)
    "gpt-5","gpt-5-mini","gpt-5-nano",
    # Gemini chat (updated models)
    "gemini-2.5-pro","gemini-2.5-flash","gemini-2.5-flash-lite",
    "gemini-2.0-flash","gemini-2.0-flash-lite",
    # Claude chat
    "claude-sonnet-4-5-20250929","claude-haiku-4-5-20251001","claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514","claude-3-7-sonnet-20250219","claude-3-5-haiku-20241022"
  ),
  provider      = c(
    rep("openai", 4),
    rep("openai", 3),
    rep("openai", 3),
    rep("gemini", 5),
    rep("claude", 6)
  ),
  type          = c(
    rep("chat", 4),
    rep("reasoning", 3),
    rep("reasoning", 3),
    rep("chat", 5),
    rep("chat", 6)
  ),
  supports_temp = c(
    rep(TRUE, 4),   # chat models use temperature
    rep(FALSE, 3),  # o-series reasoning: ignore temperature
    rep(FALSE, 3),  # GPT-5 reasoning: ignore temperature
    rep(TRUE, 5),   # gemini chat
    rep(TRUE, 6)    # claude chat
  ),
  stringsAsFactors = FALSE
)

#' Normalization map for model name aliases
#'
#' Keys are lowercased with all whitespace removed before lookup.
#' Values are canonical names as listed in MODEL_SPEC$name.
#'
#' @keywords internal
#' @noRd
NORMALIZE_MAP <- c(
  # o-series
  "01"       = "o1",
  "01-mini"  = "o1-mini",
  "03-mini"  = "o3-mini",

  # GPT-4 family
  "4o-mini"  = "gpt-4o-mini",
  "4o"       = "gpt-4o",
  "4.1"      = "gpt-4.1",
  "4.1-mini" = "gpt-4.1-mini",

  # GPT-5 family (various user-typed aliases)
  "5"         = "gpt-5",
  "5mini"     = "gpt-5-mini",
  "5-mini"    = "gpt-5-mini",
  "5nano"     = "gpt-5-nano",
  "5-nano"    = "gpt-5-nano",
  "gpt5"      = "gpt-5",
  "gpt5mini"  = "gpt-5-mini",
  "gpt5-mini" = "gpt-5-mini",
  "gpt5nano"  = "gpt-5-nano",
  "gpt5-nano" = "gpt-5-nano"
)

# -------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------

#' Coalesce character vector
#'
#' @keywords internal
#' @noRd
coalesce_chr <- function(x, y){
  if (is.null(x) || !length(x) || !nzchar(x[1])) y else x
}

#' Null coalescing operator
#'
#' @keywords internal
#' @noRd
`%null%` <- function(x, y) if (is.null(x)) y else x

#' Normalize model alias to canonical name
#'
#' Normalizes common user inputs (spaces removed, lowercased for lookup)
#' into a canonical model name present in MODEL_SPEC. If no mapping is found,
#' returns the trimmed original (case preserved).
#'
#' @keywords internal
#' @noRd
normalize_model <- function(m){
  mm  <- gsub("\\s+", "", m)
  key <- tolower(mm)
  if (key %in% names(NORMALIZE_MAP)) NORMALIZE_MAP[[key]] else mm
}

#' Get one row from MODEL_SPEC or error
#'
#' @keywords internal
#' @noRd
get_spec <- function(model){
  row <- MODEL_SPEC[MODEL_SPEC$name == model, , drop = FALSE]
  if (!nrow(row)) stop("Unsupported model: ", model, call. = FALSE)
  row
}

#' Logger factory
#'
#' @keywords internal
#' @noRd
make_logger <- function(rv, enabled){
  force(enabled)
  function(...){
    if (!enabled) return(invisible())
    msg <- paste0(format(Sys.time(), "%H:%M:%S"), " | ", paste(..., collapse = " "))
    cat(msg, "\n")
    # Use isolate to update reactive value in shiny without triggering reactivity
    shiny::isolate(rv$log <- c(rv$log, msg))
  }
}

#' Perform an httr2 request with logging and error handling
#'
#' @keywords internal
#' @noRd
perform_req <- function(req, logger){
  resp <- tryCatch(httr2::req_perform(req), error = function(e){
    logger("httr2 perform error:", e$message)
    stop(e)
  })
  status <- httr2::resp_status(resp)
  logger("HTTP status:", status, httr2::resp_status_desc(resp))
  if (status >= 400){
    body_txt <- httr2::resp_body_string(resp)
    logger("HTTP error body:", body_txt)
    stop(sprintf("HTTP %s %s", status, httr2::resp_status_desc(resp)), call. = FALSE)
  }
  resp
}

#' Strip 'A)' / 'B)' prefixes often found in LLM outputs
#'
#' @keywords internal
#' @noRd
strip_AB_prefix <- function(x){
  sub("^\\s*[A-Z]\\)\\s*", "", x, perl = TRUE)
}

#' Remove Markdown code fences from a string
#'
#' @keywords internal
#' @noRd
strip_code_fences <- function(txt){
  txt <- sub("^```[a-zA-Z0-9_-]*\\s*", "", txt)
  txt <- sub("```\\s*$", "", txt)
  trimws(txt)
}

#' Parse reconciliation JSON or fallback to first-line parsing
#'
#' @keywords internal
#' @noRd
parse_recon_output <- function(txt){
  txt2 <- strip_code_fences(txt)
  try_json <- try(jsonlite::fromJSON(txt2), silent = TRUE)
  if (!inherits(try_json, "try-error") &&
      all(c("revised","explanation") %in% names(try_json))){
    return(list(revised = try_json$revised, explanation = try_json$explanation))
  }
  parts <- strsplit(txt2, "\n", fixed = TRUE)[[1]]
  if (!length(parts)) return(list(revised = txt2, explanation = ""))
  list(
    revised     = strip_AB_prefix(parts[1]),
    explanation = if (length(parts) > 1)
      strip_AB_prefix(paste(parts[-1], collapse = " ")) else ""
  )
}

#' Parse batch translation response (forward or backward)
#'
#' Expects a JSON array with objects containing either "translation" or "back_translation"
#'
#' @keywords internal
#' @noRd
parse_batch_response <- function(txt, field_name, n_items, logger = function(...) {}){
  txt2 <- strip_code_fences(txt)

  # Try to parse as JSON
  parsed <- try(jsonlite::fromJSON(txt2, simplifyVector = FALSE), silent = TRUE)

  if (inherits(parsed, "try-error") || !is.list(parsed)) {
    logger("Failed to parse batch response as JSON. Attempting line-by-line parsing...")
    # Fallback: try to parse line by line
    lines <- strsplit(txt2, "\n", fixed = TRUE)[[1]]
    lines <- lines[nzchar(trimws(lines))]
    result <- vector("list", n_items)
    for (i in seq_len(min(n_items, length(lines)))) {
      # Remove leading numbers and dots
      cleaned <- sub("^\\s*\\d+\\.?\\s*", "", lines[i])
      result[[i]] <- cleaned
    }
    # Fill remaining with error message
    if (length(result) < n_items) {
      for (i in (length(result) + 1):n_items) {
        result[[i]] <- "ERROR: Could not parse response"
      }
    }
    return(result)
  }

  # Extract translations from parsed JSON
  # Use item_number field to map correctly, fallback to array order
  result <- vector("list", n_items)

  for (j in seq_along(parsed)) {
    # Check if item has item_number field
    item_num <- parsed[[j]][["item_number"]]

    if (!is.null(item_num)) {
      # Use the item_number specified by LLM
      idx <- as.integer(item_num)
      if (idx >= 1 && idx <= n_items) {
        result[[idx]] <- as.character(parsed[[j]][[field_name]] %null% "ERROR: Missing field")
      } else {
        logger(paste("Warning: Item number", idx, "out of range (1 to", n_items, ")"))
      }
    } else {
      # Fallback: use array position if no item_number field
      if (j <= n_items) {
        result[[j]] <- as.character(parsed[[j]][[field_name]] %null% "ERROR: Missing field")
      }
    }
  }

  # Fill any remaining nulls with error message
  for (i in seq_len(n_items)) {
    if (is.null(result[[i]])) {
      result[[i]] <- "ERROR: Missing translation in response"
      logger(paste("Warning: Missing", field_name, "for item", i))
    }
  }

  result
}

#' Parse batch reconciliation response
#'
#' Expects a JSON array with objects containing "revised" and "explanation"
#'
#' @keywords internal
#' @noRd
parse_batch_recon_response <- function(txt, n_items, logger = function(...) {}){
  txt2 <- strip_code_fences(txt)

  # Try to parse as JSON
  parsed <- try(jsonlite::fromJSON(txt2, simplifyVector = FALSE), silent = TRUE)

  if (inherits(parsed, "try-error") || !is.list(parsed)) {
    logger("Failed to parse batch reconciliation response as JSON. Using fallback...")
    # Fallback: return error for all items
    result <- vector("list", n_items)
    for (i in seq_len(n_items)) {
      result[[i]] <- list(revised = "ERROR: Could not parse response", explanation = "")
    }
    return(result)
  }

  # Extract reconciliation data from parsed JSON
  # Use item_number field to map correctly, fallback to array order
  result <- vector("list", n_items)

  for (j in seq_along(parsed)) {
    # Check if item has item_number field
    item_num <- parsed[[j]][["item_number"]]

    revised <- if (!is.null(parsed[[j]][["revised"]])) {
      as.character(parsed[[j]][["revised"]])
    } else {
      "ERROR: Missing revised field"
    }

    explanation <- if (!is.null(parsed[[j]][["explanation"]])) {
      as.character(parsed[[j]][["explanation"]])
    } else {
      ""
    }

    if (!is.null(item_num)) {
      # Use the item_number specified by LLM
      idx <- as.integer(item_num)
      if (idx >= 1 && idx <= n_items) {
        result[[idx]] <- list(revised = revised, explanation = explanation)
      } else {
        logger(paste("Warning: Item number", idx, "out of range (1 to", n_items, ")"))
      }
    } else {
      # Fallback: use array position if no item_number field
      if (j <= n_items) {
        result[[j]] <- list(revised = revised, explanation = explanation)
      }
    }
  }

  # Fill any remaining nulls with error message
  for (i in seq_len(n_items)) {
    if (is.null(result[[i]])) {
      result[[i]] <- list(revised = "ERROR: Missing item in response", explanation = "")
      logger(paste("Warning: Missing reconciliation data for item", i))
    }
  }

  result
}

# -------------------------------------------------------------------
# API wrappers
# -------------------------------------------------------------------

#' Call OpenAI chat/completions endpoint
#'
#' @keywords internal
#' @noRd
call_openai_chat <- function(model, prompt, temperature, api_key, logger){
  body <- list(
    model    = model,
    messages = list(list(role = "user", content = prompt))
  )
  if (!is.null(temperature)) body$temperature <- temperature

  logger("OpenAI chat model:", model)
  logger("Prompt(first 120):", substr(prompt, 1, 120))
  logger("Body:", jsonlite::toJSON(body, auto_unbox = TRUE))

  req  <- httr2::request("https://api.openai.com/v1/chat/completions") |>
    httr2::req_auth_bearer_token(api_key) |>
    httr2::req_body_json(body)

  dat  <- httr2::resp_body_json(perform_req(req, logger))
  if (!is.null(dat$error)) stop(dat$error$message, call. = FALSE)
  out <- dat$choices[[1]]$message$content %null% ""
  logger("OpenAI chat out(first 120):", substr(out, 1, 120))
  out
}

#' Call OpenAI reasoning 'responses' endpoint (GPT-5 and o-series)
#'
#' @keywords internal
#' @noRd
call_openai_reasoning_responses <- function(model, prompt, effort, api_key, logger){
  body <- list(
    model    = model,
    input    = list(list(
      role    = "user",
      content = list(list(type = "text", text = prompt))
    )),
    reasoning         = list(effort = effort),
    max_output_tokens = 2048
  )
  logger("OpenAI responses model:", model, "effort:", effort)
  logger("Prompt(first 120):", substr(prompt, 1, 120))
  logger("Body:", jsonlite::toJSON(body, auto_unbox = TRUE))

  req  <- httr2::request("https://api.openai.com/v1/responses") |>
    httr2::req_auth_bearer_token(api_key) |>
    httr2::req_body_json(body)

  dat  <- httr2::resp_body_json(perform_req(req, logger))
  if (!is.null(dat$error)) stop(dat$error$message, call. = FALSE)

  out <- dat$output_text %null% {
    # fallback if structure changes
    tryCatch(
      paste0(vapply(dat$output[[1]]$content, `[[`, "", "text"), collapse = "\n"),
      error = function(e) ""
    )
  }
  logger("OpenAI responses out(first 120):", substr(out, 1, 120))
  out
}

#' Call Google Gemini chat endpoint
#'
#' @keywords internal
#' @noRd
call_gemini_chat <- function(model, prompt, temperature, api_key, logger){
  endpoint_try <- function(mod){
    base <- "https://generativelanguage.googleapis.com/v1beta"
    url  <- glue::glue("{base}/models/{mod}:generateContent?key={api_key}")
    body <- list(
      contents = list(list(parts = list(list(text = prompt)))),
      generationConfig = list(temperature = temperature)
    )
    logger("Gemini model:", mod)
    logger("Prompt(first 120):", substr(prompt, 1, 120))
    logger("Body:", jsonlite::toJSON(body, auto_unbox = TRUE))

    req  <- httr2::request(url) |>
      httr2::req_method("POST") |>
      httr2::req_body_json(body)

    resp <- perform_req(req, logger)
    httr2::resp_body_json(resp)
  }

  dat <- tryCatch(endpoint_try(model), error = function(e){
    if (grepl("404", e$message) && model == "gemini-1.0-pro"){
      logger("404 for gemini-1.0-pro; falling back to gemini-2.5-flash")
      endpoint_try("gemini-2.5-flash")
    } else stop(e)
  })

  if (!is.null(dat$error)) stop(dat$error$message, call. = FALSE)
  out <- tryCatch({
    paste0(vapply(dat$candidates[[1]]$content$parts, `[[`, "", "text"), collapse = "\n")
  }, error = function(e) "")
  logger("Gemini out(first 120):", substr(out, 1, 120))
  out
}

#' Call Anthropic Claude chat endpoint
#'
#' @keywords internal
#' @noRd
call_claude_chat <- function(model, prompt, temperature, api_key, logger){
  body <- list(
    model = model,
    messages = list(list(role = "user", content = prompt)),
    max_tokens = 2048
  )
  if (!is.null(temperature)) body$temperature <- temperature

  logger("Claude chat model:", model)
  logger("Prompt(first 120):", substr(prompt, 1, 120))
  logger("Body:", jsonlite::toJSON(body, auto_unbox = TRUE))

  req <- httr2::request("https://api.anthropic.com/v1/messages") |>
    httr2::req_headers(
      `x-api-key` = api_key,
      `anthropic-version` = "2023-06-01",
      `content-type` = "application/json"
    ) |>
    httr2::req_body_json(body)

  dat <- httr2::resp_body_json(perform_req(req, logger))
  if (!is.null(dat$error)) stop(dat$error$message, call. = FALSE)

  out <- tryCatch({
    paste0(vapply(dat$content, function(x) if (!is.null(x$text)) x$text else "", ""), collapse = "\n")
  }, error = function(e) "")

  logger("Claude out(first 120):", substr(out, 1, 120))
  out
}

#' Generic LLM call dispatcher
#'
#' @keywords internal
#' @noRd
llm_call <- function(model, prompt, temperature = 0,
                     openai_key = NULL, gemini_key = NULL, claude_key = NULL,
                     logger = function(...) {}, effort = "medium"){
  spec <- get_spec(model)
  if (spec$provider == "openai"){
    key <- coalesce_chr(openai_key, Sys.getenv("OPENAI_API_KEY"))
    if (!nzchar(key)) stop("Missing OpenAI API key", call. = FALSE)

    if (spec$type == "reasoning"){
      out <- tryCatch(
        call_openai_reasoning_responses(model, prompt, effort, key, logger),
        error = function(e){
          logger("Responses failed for", model, ":", e$message, "-> fallback gpt-4.1-mini")
          call_openai_chat("gpt-4.1-mini", prompt, 0, key, logger)
        }
      )
      return(out)
    } else {
      temp <- if (spec$supports_temp) temperature else NULL
      return(call_openai_chat(model, prompt, temp, key, logger))
    }

  } else if (spec$provider == "gemini"){
    key <- coalesce_chr(gemini_key, Sys.getenv("GEMINI_API_KEY"))
    if (!nzchar(key)) stop("Missing Gemini API key", call. = FALSE)
    temp <- if (spec$supports_temp) temperature else NULL
    return(call_gemini_chat(model, prompt, temp, key, logger))

  } else if (spec$provider == "claude"){
    key <- coalesce_chr(claude_key, Sys.getenv("CLAUDE_API_KEY"))
    if (!nzchar(key)) stop("Missing Claude API key", call. = FALSE)
    temp <- if (spec$supports_temp) temperature else NULL
    return(call_claude_chat(model, prompt, temp, key, logger))

  } else {
    stop("Unsupported provider: ", spec$provider, call. = FALSE)
  }
}

# -------------------------------------------------------------------
# Optional: Avoid R CMD check "no visible binding" notes for globals
# -------------------------------------------------------------------
utils::globalVariables(c("MODEL_SPEC", "NORMALIZE_MAP"))
