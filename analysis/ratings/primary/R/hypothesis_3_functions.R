# 仮説3で使う関数 ------------------------------------------------------------
# 呼び出し側で library(tidyverse) と source("R/hypothesis_3_functions.R") をして使う
# （h3_read() と h3_prepare() は dplyr・tidyr の関数を名前空間なしで使う）。
# 関数の定義だけを置く（データの読み込み・実行・判定は h3.qmd に書く）。
#
# 【注意】h3_fit() は関数本体のコードをキャッシュのキーに含める。本体を1文字でも変えると
# results/h3/mcmc_cache/ の保存済み推定結果が使えなくなり，MCMC をやり直すことになる。
#
# 推定には cmdstanr バックエンドを使い，各チェーンの中を threading() で並列化する
# （cmdstanr と CmdStan のインストールが必要）。

# =============================================================================
# 0. β の事前分布（事前登録：Cauchy(0, √2/2)）
# =============================================================================
# 事前登録の尺度 r = √2/2 は標準化効果量 δ の単位で決められている。β は対数オッズなので，
# ロジスティック分布の潜在変数の SD（π/√3）を掛けて β の単位に直す：
#   δ = β / (π/√3) ~ Cauchy(0, r)  ⇔  β ~ Cauchy(0, r × π/√3)
# 標準化には誤差の SD だけを使い，変量効果の分散は含めない。
# 引数 prior_r はすべて δ の単位の尺度で指定する。
h3_prior_r <- sqrt(2) / 2
# 感度分析で比べる尺度（δ の単位）：狭い 1/2，事前登録の √2/2，広い 1
h3_sens_rs <- c(0.5, sqrt(2) / 2, 1)

# β の単位の尺度。Stan に渡す文字列と密度の計算で同じ値になるよう小数6桁に丸める。
h3_beta_scale <- function(r) round(r * pi / sqrt(3), 6)
h3_beta_prior <- function(r) sprintf("cauchy(0, %.6f)", h3_beta_scale(r))
h3_prior_density <- function(beta, r) stats::dcauchy(beta, 0, h3_beta_scale(r))

# 浮動小数の r を比べる（sqrt(2) / 2 などを %in% で探すと一致しないことがある）。
h3_r_in <- function(x, r) vapply(x, function(v) any(abs(v - r) < 1e-8), logical(1))

# =============================================================================
# 1. データの読み込みと前処理
# =============================================================================
h3_read <- function() {
  h3_wide <- setNames(lapply(LETTERS[1:6], function(s) {
    readr::read_csv(
      paste0("data/3_data_processed/ManyScales_市民_processed_", s, ".csv"),
      show_col_types = FALSE
    )
  }), LETTERS[1:6])
  expected_ids <- list(A = 1L, B = 2:5, C = 6:10, D = 10:14, E = 15:18, F = 19:22)
  purrr::imap_dfr(h3_wide, function(w, s) {
    expected_cols <- as.vector(outer(
      sprintf("%s%02d_wording_", s, expected_ids[[s]]), c("human", "ai"), paste0
    ))
    actual_cols <- grep("^[A-F][0-9]{2}_wording_(human|ai)$", names(w), value = TRUE)
    stopifnot(setequal(actual_cols, expected_cols),
              all(c("gender", "trans_cond") %in% names(w)),
              !anyNA(w$trans_cond), all(w$trans_cond %in% c("human", "ai")))
    w |>
      mutate(set = s, subject_id = sprintf("%s_%03d", s, row_number())) |>
      select(set, subject_id, gender, trans_cond, all_of(expected_cols)) |>
      pivot_longer(
        cols = all_of(expected_cols),
        names_to = c("scale_id", "version"),
        names_pattern = "^[A-F]([0-9]{2})_wording_(human|ai)$",
        values_to = "response"
      ) |>
      mutate(scale_id = as.integer(scale_id))
  })
}

h3_prepare <- function(raw, high_is_good = TRUE, complete_pairs = FALSE) {
  stopifnot(is.logical(high_is_good), length(high_is_good) == 1L, !is.na(high_is_good),
            !anyNA(raw[c("subject_id", "scale_id", "version")]),
            !anyDuplicated(raw[c("subject_id", "scale_id", "version")]),
            all(raw$version %in% c("human", "ai")),
            all(is.na(raw$response) | raw$response %in% 1:5))
  if (any(raw$scale_id == 10L & is.na(raw$gender))) {
    stop("MCS対象者の性別に欠測があります。適用条件を確認してください。")
  }
  noneligible <- raw$scale_id == 10L & !is.na(raw$gender) & raw$gender != 1
  if (any(!is.na(raw$response[noneligible]))) {
    stop("MCS非該当者にwording評定があります。調査設計とデータを確認してください。")
  }
  d <- raw[!noneligible, ]
  pair_counts <- d |> count(subject_id, scale_id)
  stopifnot(all(pair_counts$n == 2L))
  d <- d |>
    mutate(score = if (high_is_good) response else 6 - response,
           pair_id = interaction(subject_id, scale_id, drop = TRUE),
           x = if_else(version == "human", 0.5, -0.5),
           period_c = if_else(version == trans_cond, -0.5, 0.5),
           human_first_c = if_else(trans_cond == "human", 0.5, -0.5))
  if (complete_pairs) {
    d <- d |> group_by(pair_id) |> filter(all(!is.na(score))) |> ungroup()
  }
  d |>
    filter(!is.na(score)) |>
    mutate(rating = ordered(score, levels = 1:5),
           subject_id = factor(subject_id), scale_id = factor(scale_id),
           pair_id = droplevels(pair_id))
}

# =============================================================================
# 2. 推定・集計・診断
# =============================================================================
h3_cached <- function(kind, signature, compute, force_refit = FALSE) {
  stopifnot(requireNamespace("digest", quietly = TRUE))
  signature$brms_version <- as.character(utils::packageVersion("brms"))
  key <- digest::digest(signature, algo = "sha256")
  cache_dir <- "results/h3/mcmc_cache"
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(cache_dir, paste0(kind, "_", key, ".rds"))
  if (!force_refit && file.exists(path)) {
    message("保存済みの推定結果を使用: ", path)
    return(readRDS(path))
  }
  value <- compute()
  temporary <- tempfile(pattern = "h3_", tmpdir = cache_dir)
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(value, temporary)
  if (!file.rename(temporary, path)) stop("キャッシュの保存に失敗しました: ", path)
  value
}

# 1チェーンあたりのスレッド数：物理コア数を4チェーンで分ける（最低1）。
h3_threads_per_chain <- function(chains = 4) {
  n <- parallel::detectCores(logical = FALSE)
  if (is.na(n)) n <- parallel::detectCores()
  max(1L, as.integer(floor(n / chains)))
}

# スレッド数は計算の速さだけに関わるので，キャッシュのキーには含めない。
h3_fit <- function(d, prior_only = FALSE, prior_r = h3_prior_r,
                   adjust_order = FALSE, threads_per_chain = h3_threads_per_chain(),
                   force_refit = FALSE) {
  stopifnot(requireNamespace("brms", quietly = TRUE),
            requireNamespace("cmdstanr", quietly = TRUE))
  dd <- d
  model_formula <- if (adjust_order) {
    rating ~ x + period_c + human_first_c +
      (1 + x || subject_id) + (1 + x || scale_id) + (1 | pair_id)
  } else {
    rating ~ x + (1 + x || subject_id) + (1 + x || scale_id) + (1 | pair_id)
  }
  h3_cached("fit", signature = list(
    data = dd, prior_only = prior_only,
    beta_prior = h3_beta_prior(prior_r), adjust_order = adjust_order,
    backend = "cmdstanr",
    code = paste(deparse(body(h3_fit)), collapse = "\n")
  ), force_refit = force_refit, compute = function() brms::brm(
    formula = model_formula,
    data = dd, family = brms::cumulative("logit"),
    prior = c(brms::set_prior(h3_beta_prior(prior_r), class = "b"),
              brms::set_prior("normal(0, 2.5)", class = "Intercept"),
              brms::set_prior("normal(0, 1)", class = "sd")),
    # Cauchy の事前サンプルは裾が重く hypothesis() の密度推定に向かないので保存しない
    sample_prior = if (prior_only) "only" else "no",
    chains = 4, cores = 4, iter = 4000, warmup = 2000, seed = 20261004,
    control = list(adapt_delta = 0.99, max_treedepth = 15),
    backend = "cmdstanr", threads = brms::threading(threads_per_chain),
    save_pars = brms::save_pars(all = TRUE)
  ))
}

h3_summary <- function(z) {
  out <- c(mean = mean(z), median = median(z),
           lower95 = unname(quantile(z, .025)), upper95 = unname(quantile(z, .975)),
           Pr_human_better = mean(z > 0))
  out
}

h3_results <- function(fit) {
  if ("period_c" %in% rownames(brms::fixef(fit))) {
    stop("h3_results()は主解析用です。提示順調整モデルは追加解析のコードで比較してください。")
  }
  draws <- posterior::as_draws_df(fit)
  scale_coef <- stats::coef(fit, summary = FALSE)$scale_id[, , "x"]
  nd <- unique(fit$data[c("subject_id", "scale_id", "pair_id")])
  ids <- split(seq_len(nrow(nd)), nd$scale_id, drop = TRUE)
  by_scale <- vapply(ids, function(j) {
    human <- ai <- nd[j, , drop = FALSE]
    human$x <- 0.5
    ai$x <- -0.5
    ph <- brms::posterior_epred(fit, newdata = human, re_formula = NULL)
    pa <- brms::posterior_epred(fit, newdata = ai, re_formula = NULL)
    stopifnot(length(dim(ph)) == 3L, dim(ph)[3] == 5L)
    delta <- matrix(0, nrow = dim(ph)[1], ncol = nrow(human))
    for (k in 1:5) {
      delta <- delta + k * (matrix(ph[, , k], nrow = nrow(delta)) -
                            matrix(pa[, , k], nrow = nrow(delta)))
    }
    rowMeans(delta)
  }, numeric(nrow(draws)))
  list(
    population_log_odds = h3_summary(draws$b_x),
    population_odds_ratio = quantile(exp(draws$b_x), c(.025, .5, .975)),
    scale_log_odds = t(apply(scale_coef, 2, h3_summary)),
    scale_points = t(apply(by_scale, 2, h3_summary)),
    equal_scale_average_points = h3_summary(rowMeans(by_scale)),
    scale_heterogeneity = quantile(draws$sd_scale_id__x, c(.025, .5, .975))
  )
}

h3_diagnostics <- function(fit) {
  s <- posterior::summarise_draws(posterior::as_draws_array(fit))
  np <- brms::nuts_params(fit)
  list(parameters = s,
       divergences = sum(np$Parameter == "divergent__" & np$Value > 0),
       treedepth_hits = sum(np$Parameter == "treedepth__" & np$Value >= 15))
}

# =============================================================================
# 3. 結果の整理・Savage–Dickey 法・感度分析・判定
# =============================================================================

# 収束診断の要約表 ------------------------------------------------------------
# h3_diagnostics() は全パラメータ（約6700行）を返すので，判断に使う値だけを1行にまとめる。
h3_diag_table <- function(fits) {
  rows <- lapply(names(fits), function(m) {
    dg <- h3_diagnostics(fits[[m]])
    s <- dg$parameters[!is.na(dg$parameters$rhat), ]
    data.frame(
      model = m,
      max_rhat = max(s$rhat),
      min_ess_bulk = min(s$ess_bulk),
      min_ess_tail = min(s$ess_tail),
      divergences = dg$divergences,
      treedepth_hits = dg$treedepth_hits
    )
  })
  out <- do.call(rbind, rows)
  out$ok <- out$max_rhat < 1.01 & out$min_ess_bulk > 400 & out$min_ess_tail > 400 &
    out$divergences == 0 & out$treedepth_hits == 0
  out
}

# サンプリングの効率（1回の反復あたりの leapfrog 数と tree depth）。
# 計算時間の内訳を確かめるために使う。
h3_sampler_table <- function(fit) {
  np <- brms::nuts_params(fit)
  get <- function(p) np$Value[np$Parameter == p]
  chain <- np$Chain[np$Parameter == "n_leapfrog__"]
  data.frame(
    chain = sort(unique(chain)),
    mean_leapfrog = tapply(get("n_leapfrog__"), chain, mean),
    max_treedepth = tapply(get("treedepth__"), chain, max),
    mean_stepsize = tapply(get("stepsize__"), chain, mean),
    row.names = NULL
  )
}

# h3_results() の全体の結果を1行にまとめる。
h3_population_table <- function(r) {
  data.frame(
    beta_mean = r$population_log_odds[["mean"]],
    beta_lower95 = r$population_log_odds[["lower95"]],
    beta_upper95 = r$population_log_odds[["upper95"]],
    pr_human_better = r$population_log_odds[["Pr_human_better"]],
    points_mean = r$equal_scale_average_points[["mean"]],
    points_lower95 = r$equal_scale_average_points[["lower95"]],
    points_upper95 = r$equal_scale_average_points[["upper95"]],
    scale_sd_median = r$scale_heterogeneity[["50%"]],
    scale_sd_lower95 = r$scale_heterogeneity[["2.5%"]],
    scale_sd_upper95 = r$scale_heterogeneity[["97.5%"]]
  )
}

# 尺度別の効果を縦長にまとめる（フォレストプロット用）。
# type = "scale_log_odds"（対数オッズ）または "scale_points"（評定点）。
h3_scale_table <- function(r, type = c("scale_log_odds", "scale_points")) {
  type <- match.arg(type)
  m <- r[[type]]
  data.frame(scale_id = as.integer(rownames(m)),
             mean = m[, "mean"], lower95 = m[, "lower95"],
             upper95 = m[, "upper95"], row.names = NULL)
}

# ベイズファクター（Savage–Dickey法） -----------------------------------------
# H0: beta = 0 は H1: beta ~ Cauchy(0, r × π/√3) の beta を 0 に固定した入れ子モデルであり，
# beta 以外のパラメータの事前分布は両モデルで同じ（beta とも独立）。このとき
#   BF01 = p(beta = 0 | データ, H1) / p(beta = 0 | H1)
# が成り立つ（Savage–Dickey 密度比）。分母は Cauchy 分布なので解析的に計算し，
# 分子は beta の事後サンプルから 0 における密度を推定する。
# H0 のモデルを別に推定する必要がない。

# 事後サンプル z の点 at における密度（カーネル密度推定）。
h3_density_at <- function(z, at = 0) {
  dens <- stats::density(z, n = 2^12, from = min(z, at) - 0.5, to = max(z, at) + 0.5)
  stats::approx(dens$x, dens$y, xout = at)$y
}

# 証拠の強さの言葉。bf は「favor のモデル」を支持する BF（例：BF01 なら平均差0）。
h3_bf_label <- function(bf, favor, against) {
  ifelse(bf >= 10, paste0(favor, "を強く支持"),
  ifelse(bf >= 3, paste0(favor, "を支持"),
  ifelse(bf > 1 / 3, "判断保留",
  ifelse(bf > 1 / 10, paste0(against, "を支持"), paste0(against, "を強く支持")))))
}

# 主解析モデルから両側・片側の BF をまとめて計算する。
#   BF10      : 平均差あり（両方向） vs 平均差0
#   BF_plus0  : 人手版が高い（beta > 0） vs 平均差0
#   BF_minus0 : AI版が高い（beta < 0） vs 平均差0
# 片側 BF は BF10 × Pr(方向 | データ) / Pr(方向 | 事前) で求める（Morey & Wagenmakers, 2014）。
# brms::hypothesis() は事前サンプルの密度推定を使うが，Cauchy は裾が重く 0 での密度が
# 不安定になるので，確認には使わない（分母は解析的に求まる）。
h3_savage_dickey <- function(fit, prior_r = h3_prior_r) {
  prior_b <- fit$prior$prior[fit$prior$class == "b" & fit$prior$coef == ""]
  stopifnot(identical(prior_b, h3_beta_prior(prior_r)))
  draws <- posterior::as_draws_df(fit)
  b <- draws$b_x
  prior0 <- h3_prior_density(0, prior_r)
  bf01 <- h3_density_at(b, 0) / prior0
  pr_pos <- mean(b > 0)
  # 計算の安定性の確認：(1) 正規近似，(2) チェーンごと
  bf01_normal <- stats::dnorm(0, mean(b), stats::sd(b)) / prior0
  bf01_chain <- tapply(b, draws$.chain, function(z) h3_density_at(z, 0) / prior0)
  data.frame(
    prior_r = prior_r, BF01 = bf01, BF10 = 1 / bf01,
    BF_plus0 = (1 / bf01) * pr_pos / 0.5,
    BF_minus0 = (1 / bf01) * (1 - pr_pos) / 0.5,
    pr_human_better = pr_pos,
    BF01_normal_approx = bf01_normal,
    BF01_chain_min = min(bf01_chain), BF01_chain_max = max(bf01_chain)
  )
}

# 事前分布の感度分析（再推定なし） --------------------------------------------
# H1 の beta の事前分布だけを尺度 base_r の Cauchy から尺度 r の Cauchy に変えた場合の BF10 は
#   BF10(r) = BF10(base_r) × E_post[ Cauchy(beta; r) / Cauchy(beta; base_r) ]
# で厳密に求まる（期待値は base_r で推定した事後分布に関するもの）。H0 は beta を
# 含まないので変わらない。重みの比は max(r / base_r, base_r / r) 以下に収まるが，
# 重みの有効サンプルサイズ ess_w が小さい（目安 < 1000）r では再推定で確かめる。
h3_bf_prior_curve <- function(fit, bf10_base, base_r = h3_prior_r,
                              prior_rs = sort(unique(c(seq(0.1, 1.5, by = 0.05),
                                                       h3_prior_r)))) {
  b <- posterior::as_draws_df(fit)$b_x
  do.call(rbind, lapply(prior_rs, function(r) {
    w <- h3_prior_density(b, r) / h3_prior_density(b, base_r)
    bf10 <- bf10_base * mean(w)
    pr_pos <- sum(w[b > 0]) / sum(w)
    data.frame(prior_r = r, BF10 = bf10, BF01 = 1 / bf10,
               BF_plus0 = bf10 * pr_pos / 0.5, BF_minus0 = bf10 * (1 - pr_pos) / 0.5,
               ess_w = sum(w)^2 / sum(w^2))
  }))
}

# ブリッジサンプリングの反復（任意） --------------------------------------------
# H1・H0 を事後サンプルを増やして別々に推定し，乱数シードを変えて bridge_sampler() を
# 繰り返す。仮説3のモデルはパラメータが約6700個あるため非常に時間とメモリを使う。
# キャッシュのキーには brmsfit を直接入れず，推定の入力（データ・式・設定）を入れる。
# 主解析と同じく cmdstanr バックエンドで推定し，チェーン内を threading() で並列化する。

# H0 は beta を含まないので事前分布を変えても同じモデルになる。保存済みの H0 の推定結果と
# ブリッジサンプリングの結果を使い続けられるよう，H0 のキーは正規事前分布のときと同じ形
# （slope_prior_sd = NA）のままにし，H1 だけ事前分布の文字列をキーに入れる。
h3_fit_bf_signature <- function(d, null_model, prior_r, iter, warmup) {
  list(data = d, null_model = null_model,
       slope_prior_sd = if (null_model) NA else h3_beta_prior(prior_r),
       iter = iter, warmup = warmup, backend = "cmdstanr")
}

h3_fit_bf <- function(d, null_model = FALSE, prior_r = h3_prior_r,
                      iter = 12000, warmup = 2000,
                      threads_per_chain = h3_threads_per_chain(), force_refit = FALSE) {
  stopifnot(requireNamespace("brms", quietly = TRUE),
            requireNamespace("cmdstanr", quietly = TRUE))
  random <- "(1 + x || subject_id) + (1 + x || scale_id) + (1 | pair_id)"
  model_formula <- stats::as.formula(paste(
    if (null_model) "rating ~ 1 +" else "rating ~ x +", random))
  priors <- c(brms::set_prior("normal(0, 2.5)", class = "Intercept"),
              brms::set_prior("normal(0, 1)", class = "sd"))
  if (!null_model) {
    priors <- c(priors, brms::set_prior(h3_beta_prior(prior_r), class = "b"))
  }
  h3_cached(if (null_model) "bf_null" else "bf_full",
            signature = h3_fit_bf_signature(d, null_model, prior_r, iter, warmup),
            force_refit = force_refit, compute = function() brms::brm(
    formula = model_formula, data = d, family = brms::cumulative("logit"),
    prior = priors, sample_prior = "no",
    chains = 4, cores = 4, iter = iter, warmup = warmup, seed = 20261006,
    control = list(adapt_delta = 0.99, max_treedepth = 15),
    backend = "cmdstanr", threads = brms::threading(threads_per_chain),
    save_pars = brms::save_pars(all = TRUE)
  ))
}

# 1つのモデルについて bridge_sampler() を seeds の数だけ繰り返し，対数周辺尤度を返す。
h3_bridge_repeat <- function(fit, signature, seeds = 1:5, maxiter = 5000,
                             method = "normal", force_refit = FALSE) {
  stopifnot(requireNamespace("bridgesampling", quietly = TRUE),
            requireNamespace("rstan", quietly = TRUE),
            isTRUE(getOption("brms.normalize", TRUE)))
  # cmdstanr の推定結果に bridge_sampler() を使うには，RStan でコンパイルしたモデルが
  # 必要になる（事後サンプルはそのまま使い，MCMC はやり直さない）。保存済みの結果を
  # 読み込んだ fit にはコンパイル済みモデルがないので，未計算のシードがあるときだけ
  # add_rstan_model() を1回だけ呼び，以降のシードで使い回す。
  fit_rstan <- NULL
  do.call(rbind, lapply(seeds, function(seed) {
    h3_cached("bridge", signature = list(fit = signature, seed = seed,
                                         maxiter = maxiter, method = method),
              force_refit = force_refit, compute = function() {
      if (is.null(fit_rstan)) fit_rstan <<- brms::add_rstan_model(fit)
      warns <- character(0)
      set.seed(seed)
      b <- withCallingHandlers(
        brms::bridge_sampler(fit_rstan, maxiter = maxiter, method = method, silent = TRUE),
        warning = function(w) {
          warns <<- c(warns, conditionMessage(w))
          invokeRestart("muffleWarning")
        })
      data.frame(seed = seed, logml = b$logml, niter = b$niter,
                 percentage_error = bridgesampling::error_measures(b)$percentage,
                 n_warnings = length(warns),
                 warnings = paste(unique(warns), collapse = " / "))
    })
  }))
}

h3_bridge_check <- function(d, seeds = 1:5, iter = 12000, warmup = 2000, maxiter = 5000) {
  fit1 <- h3_fit_bf(d, null_model = FALSE, iter = iter, warmup = warmup)
  fit0 <- h3_fit_bf(d, null_model = TRUE, iter = iter, warmup = warmup)
  b1 <- h3_bridge_repeat(fit1, h3_fit_bf_signature(d, FALSE, h3_prior_r, iter, warmup),
                         seeds = seeds, maxiter = maxiter)
  b0 <- h3_bridge_repeat(fit0, h3_fit_bf_signature(d, TRUE, h3_prior_r, iter, warmup),
                         seeds = seeds, maxiter = maxiter)
  bridge <- data.frame(
    seed = seeds, logml_h1 = b1$logml, logml_h0 = b0$logml,
    BF10 = exp(b1$logml - b0$logml),
    error_h1 = b1$percentage_error, error_h0 = b0$percentage_error,
    n_warnings = b1$n_warnings + b0$n_warnings
  )
  b <- posterior::as_draws_df(fit1)$b_x
  list(bridge = bridge,
       BF10_savage_dickey = h3_prior_density(0, h3_prior_r) / h3_density_at(b, 0),
       diagnostics = h3_diag_table(list(H1 = fit1, H0 = fit0)))
}

# 仮説に対する判定 ------------------------------------------------------------
# 予測は「差なし」なので BF01 で判定する（本文「判定の規則」と対応）。
# sens は h3_bf_prior_curve() の結果（prior_r = h3_sens_rs を含む）。
h3_verdict <- function(pop, sd_bf, sens) {
  ss <- sens[h3_r_in(sens$prior_r, h3_sens_rs), ]
  labels <- h3_bf_label(ss$BF01, "平均差0", "平均差あり")
  zone <- cut(ss$BF01, c(0, 1 / 3, 3, Inf))
  data.frame(
    prediction = "差なし",
    beta = sprintf("%.2f [%.2f, %.2f]", pop$beta_mean, pop$beta_lower95, pop$beta_upper95),
    pr_human_better = sprintf("%.1f%%", 100 * pop$pr_human_better),
    ci = if (pop$beta_lower95 > 0) "0より上（人手版が高い）"
         else if (pop$beta_upper95 < 0) "0より下（AI版が高い）"
         else "0を含む",
    bayes_factor = sprintf("BF01 = %.3g（BF10 = %.3g）", sd_bf$BF01, sd_bf$BF10),
    bf_verdict = h3_bf_label(sd_bf$BF01, "平均差0", "平均差あり"),
    prior_robust = if (length(unique(zone)) == 1) "r = 1/2〜1で同じ区分" else
      paste0("r で区分が変わる（", paste(unique(labels), collapse = "／"), "）"),
    points = sprintf("%+.2f [%+.2f, %+.2f]", pop$points_mean, pop$points_lower95,
                     pop$points_upper95),
    scale_sd = sprintf("%.2f [%.2f, %.2f]", pop$scale_sd_median, pop$scale_sd_lower95,
                       pop$scale_sd_upper95)
  )
}
