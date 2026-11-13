# tests/testthat/test-internals.R

test_that("normalize_model maps aliases correctly", {
  nm <- getFromNamespace("normalize_model", "LLMTranslate")
  expect_equal(nm("4o"), "gpt-4o")
  expect_equal(nm(" 4.1-mini "), "gpt-4.1-mini")

  # GPT-5 aliases
  expect_equal(nm("5"), "gpt-5")
  expect_equal(nm("5-mini"), "gpt-5-mini")
  expect_equal(nm("5mini"), "gpt-5-mini")
  expect_equal(nm("5-nano"), "gpt-5-nano")
  expect_equal(nm("5nano"), "gpt-5-nano")
  expect_equal(nm("gpt5"), "gpt-5")
  expect_equal(nm("gpt5-mini"), "gpt-5-mini")
  expect_equal(nm("gpt5-nano"), "gpt-5-nano")

  # Unknown model: returned trimmed original (case preserved)
  expect_equal(nm("SomethingElse"), "SomethingElse")
})

test_that("get_spec returns a row or errors for unsupported models", {
  gs <- getFromNamespace("get_spec", "LLMTranslate")
  expect_s3_class(gs("gpt-4o-mini"), "data.frame")

  # GPT-5 family present
  expect_s3_class(gs("gpt-5"), "data.frame")
  expect_s3_class(gs("gpt-5-mini"), "data.frame")
  expect_s3_class(gs("gpt-5-nano"), "data.frame")

  expect_error(gs("foobar-model"), "Unsupported model")
})

test_that("MODEL_SPEC has required columns", {
  ms <- getFromNamespace("MODEL_SPEC", "LLMTranslate")
  expect_true(all(c("name", "provider", "type", "supports_temp") %in% names(ms)))
  expect_true(nrow(ms) >= 1)
})

test_that("coalesce_chr returns fallback only when needed", {
  co <- getFromNamespace("coalesce_chr", "LLMTranslate")
  expect_equal(co(NULL, "b"), "b")
  expect_equal(co(character(0), "b"), "b")
  expect_equal(co("", "b"), "b")
  expect_equal(co("a", "b"), "a")
})

test_that("%null% operator works as intended", {
  nullop <- getFromNamespace("%null%", "LLMTranslate")
  expect_equal(nullop(NULL, "x"), "x")
  expect_equal(nullop("y", "x"), "y")
})

test_that("strip_code_fences removes markdown fences", {
  scf <- getFromNamespace("strip_code_fences", "LLMTranslate")
  txt <- "```json\n{\"a\":1}\n```"
  expect_equal(scf(txt), "{\"a\":1}")
  expect_equal(scf("no fences"), "no fences")
})

test_that("strip_AB_prefix removes leading letter choices", {
  sap <- getFromNamespace("strip_AB_prefix", "LLMTranslate")
  expect_equal(sap("A) Hello"), "Hello")
  expect_equal(sap("   B)  World"), "World")
  expect_equal(sap("No prefix"), "No prefix")
})

test_that("parse_recon_output handles JSON correctly", {
  pr <- getFromNamespace("parse_recon_output", "LLMTranslate")
  json_in <- '{"revised":"Hallo","explanation":"Minor tweak"}'
  out <- pr(json_in)
  expect_equal(out$revised, "Hallo")
  expect_match(out$explanation, "Minor")
})

test_that("parse_recon_output falls back when JSON missing", {
  pr <- getFromNamespace("parse_recon_output", "LLMTranslate")
  lines_in <- "Hallo\nMinor tweak"
  out2 <- pr(lines_in)
  expect_equal(out2$revised, "Hallo")
  expect_match(out2$explanation, "Minor")
})
