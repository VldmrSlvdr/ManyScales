# 仮説1で使う関数 ------------------------------------------------------------
# 呼び出し側で source("R/hypothesis_1_functions.R") して使う。
# 関数の定義だけを置く（データの読み込み・実行・判定は h1.qmd に書く）。
#
# 【注意】h1_fit() は関数本体のコードをキャッシュのキーに含める。本体を1文字でも変えると
# results/h1/mcmc_cache/ の保存済み推定結果が使えなくなり，MCMC をやり直すことになる。
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
h1_prior_r <- sqrt(2) / 2
# 感度分析で比べる尺度（δ の単位）：狭い 1/2，事前登録の √2/2，広い 1
h1_sens_rs <- c(0.5, sqrt(2) / 2, 1)

# β の単位の尺度。Stan に渡す文字列と密度の計算で同じ値になるよう小数6桁に丸める。
h1_beta_scale <- function(r) round(r * pi / sqrt(3), 6)
h1_beta_prior <- function(r) sprintf("cauchy(0, %.6f)", h1_beta_scale(r))
h1_prior_density <- function(beta, r) stats::dcauchy(beta, 0, h1_beta_scale(r))

# 浮動小数の r を比べる（sqrt(2) / 2 などを %in% で探すと一致しないことがある）。
h1_r_in <- function(x, r) vapply(x, function(v) any(abs(v - r) < 1e-8), logical(1))

# =============================================================================
# 1. データの読み込みと前処理
# =============================================================================
h1_read <- function() {
  aspects <- c("semantic", "naturalness", "cultural_validity")
  expected_ids <- list(A = 1L, B = 2:5, C = 6:10, D = 10:14, E = 15:18, F = 19:22)
  parts <- lapply(names(expected_ids), function(set_name) {
    w <- read.csv(
      paste0("data/3_data_processed/ManyScales_専門家_processed_", set_name, ".csv"),
      check.names = FALSE
    )
    stopifnot("trans_cond" %in% names(w), !anyNA(w$trans_cond),
              all(w$trans_cond %in% c("human", "ai")))
    # 元ファイルの行順のままIDを付与する（抽出・欠測除外より前）。
    subject_id <- sprintf("pro_%s_%03d", set_name, seq_len(nrow(w)))
    spec <- expand.grid(scale_id = expected_ids[[set_name]],
                        aspect = aspects, version = c("human", "ai"),
                        stringsAsFactors = FALSE)
    spec$column <- sprintf("%s%02d_%s_%s", set_name,
                           spec$scale_id, spec$aspect, spec$version)
    actual <- grep(
      "^[A-F][0-9]{2}_(semantic|naturalness|cultural_validity)_(human|ai)$",
      names(w), value = TRUE
    )
    stopifnot(setequal(actual, spec$column))
    do.call(rbind, lapply(seq_len(nrow(spec)), function(j) {
      data.frame(
        set = set_name, subject_id = subject_id,
        trans_cond = w$trans_cond, scale_id = spec$scale_id[j],
        aspect = spec$aspect[j], version = spec$version[j],
        response = w[[spec$column[j]]], stringsAsFactors = FALSE
      )
    }))
  })
  out <- do.call(rbind, parts)
  rownames(out) <- NULL
  key <- c("subject_id", "scale_id", "version", "aspect")
  stopifnot(!anyNA(out[key]), !anyDuplicated(out[key]),
            is.numeric(out$response),
            all(is.na(out$response) | out$response %in% 1:5))
  out
}

h1_prepare <- function(raw, high_is_good, complete_pairs = FALSE) {
  aspects <- c("semantic", "naturalness", "cultural_validity")
  # 処理済みデータでは文化的妥当性も逆転済みなので、3観点とも TRUE（5_仮説1_v2.qmd）。
  stopifnot(all(aspects %in% names(high_is_good)),
            is.logical(high_is_good), !anyNA(high_is_good[aspects]))
  d <- raw[raw$aspect %in% aspects, ]
  # responseを保持し、high_is_good が FALSE の観点だけ逆転した解析用得点を作る。
  d$score <- ifelse(high_is_good[d$aspect], d$response, 6 - d$response)
  d$pair_id <- interaction(d$subject_id, d$scale_id, drop = TRUE)
  d$x <- ifelse(d$version == "human", 0.5, -0.5) # 係数は人手 − AI
  # 提示順調整を行う感度分析用。主解析のモデル式には含めない。
  d$period_c <- ifelse(d$version == d$trans_cond, -0.5, 0.5)
  d$human_first_c <- ifelse(d$trans_cond == "human", 0.5, -0.5)
  if (complete_pairs) {
    k <- interaction(d$pair_id, d$aspect, drop = TRUE)
    n <- ave(as.integer(!is.na(d$score)), k, FUN = sum)
    d <- d[n == 2L, ]
  }
  d <- d[!is.na(d$score), ]
  d$rating <- ordered(d$score, levels = 1:5)
  d$subject_id <- factor(d$subject_id)
  d$scale_id <- factor(d$scale_id)
  d
}

# =============================================================================
# 2. 推定・集計・診断
# =============================================================================
h1_cached <- function(kind, signature, compute, force_refit = FALSE) {
  stopifnot(requireNamespace("digest", quietly = TRUE))
  signature$brms_version <- as.character(utils::packageVersion("brms"))
  key <- digest::digest(signature, algo = "sha256")
  cache_dir <- "results/h1/mcmc_cache"
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(cache_dir, paste0(kind, "_", key, ".rds"))
  if (!force_refit && file.exists(path)) {
    message("保存済みの推定結果を使用: ", path)
    return(readRDS(path))
  }
  value <- compute()
  temporary <- tempfile(pattern = "h1_", tmpdir = cache_dir)
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(value, temporary)
  if (!file.rename(temporary, path)) stop("キャッシュの保存に失敗しました: ", path)
  value
}

# 1チェーンあたりのスレッド数：物理コア数を4チェーンで分ける（最低1）。
h1_threads_per_chain <- function(chains = 4) {
  n <- parallel::detectCores(logical = FALSE)
  if (is.na(n)) n <- parallel::detectCores()
  max(1L, as.integer(floor(n / chains)))
}

# スレッド数は計算の速さだけに関わるので，キャッシュのキーには含めない。
h1_fit <- function(d, aspect_name, prior_only = FALSE, prior_r = h1_prior_r,
                   adjust_order = FALSE, threads_per_chain = h1_threads_per_chain(),
                   force_refit = FALSE) {
  stopifnot(requireNamespace("brms", quietly = TRUE),
            requireNamespace("cmdstanr", quietly = TRUE))
  dd <- droplevels(d[d$aspect == aspect_name, ])
  model_formula <- if (adjust_order) {
    rating ~ x + period_c + human_first_c +
      (1 + x || subject_id) + (1 + x || scale_id) + (1 | pair_id)
  } else {
    rating ~ x + (1 + x || subject_id) + (1 + x || scale_id) + (1 | pair_id)
  }
  h1_cached("fit", signature = list(
    data = dd, aspect = aspect_name, prior_only = prior_only,
    beta_prior = h1_beta_prior(prior_r), adjust_order = adjust_order,
    backend = "cmdstanr",
    code = paste(deparse(body(h1_fit)), collapse = "\n")
  ), force_refit = force_refit, compute = function() brms::brm(
    formula = model_formula,
    data = dd, family = brms::cumulative("logit"),
    prior = c(brms::set_prior(h1_beta_prior(prior_r), class = "b"),
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

h1_summary <- function(z) {
  out <- c(mean = mean(z), median = median(z),
           lower95 = unname(quantile(z, .025)), upper95 = unname(quantile(z, .975)),
           Pr_human_better = mean(z > 0))
  out
}

h1_results <- function(fit) {
  if ("period_c" %in% names(brms::fixef(fit)[, "Estimate"])) {
    stop("h1_results()は主解析用です。提示順調整モデルは追加解析のコードで比較してください。")
  }
  draws <- posterior::as_draws_df(fit)
  # 尺度母集団の平均効果（条件付き対数オッズ）。点数差とは異なる。
  population <- h1_summary(draws$b_x)
  scale_coef <- stats::coef(fit, summary = FALSE)$scale_id[, , "x"]
  scale_log_odds <- t(apply(scale_coef, 2, h1_summary))

  # 実際の専門家構成に標準化した予測評定差。
  # 各人×尺度に両版を予測し、尺度内平均後に各尺度を等重みで平均する。
  nd <- unique(fit$data[c("subject_id", "scale_id", "pair_id")])
  human <- ai <- nd
  human$x <- 0.5
  ai$x <- -0.5
  ph <- brms::posterior_epred(fit, newdata = human, re_formula = NULL)
  pa <- brms::posterior_epred(fit, newdata = ai, re_formula = NULL)
  stopifnot(length(dim(ph)) == 3L, dim(ph)[3] == 5L)
  delta <- matrix(0, nrow = dim(ph)[1], ncol = nrow(nd))
  for (k in 1:5) delta <- delta + k * (ph[, , k] - pa[, , k])
  by_scale <- vapply(split(seq_len(nrow(nd)), nd$scale_id, drop = TRUE),
                     function(j) rowMeans(delta[, j, drop = FALSE]),
                     numeric(nrow(delta)))
  list(population_log_odds = population,
       population_odds_ratio = quantile(exp(draws$b_x), c(.025, .5, .975)),
       scale_log_odds = scale_log_odds,
       scale_points = t(apply(by_scale, 2, h1_summary)),
       equal_scale_average_points = h1_summary(rowMeans(by_scale)),
       scale_heterogeneity = quantile(draws$sd_scale_id__x, c(.025, .5, .975)))
}

h1_diagnostics <- function(fit) {
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
# h1_diagnostics() は全パラメータ（約1300行）を返すので，判断に使う値だけを1行にまとめる。
h1_diag_table <- function(fits) {
  rows <- lapply(names(fits), function(a) {
    dg <- h1_diagnostics(fits[[a]])
    s <- dg$parameters[!is.na(dg$parameters$rhat), ]
    data.frame(
      aspect = a,
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

# 事後分布の取り出しと表 ------------------------------------------------------
# 平均翻訳効果 beta（b_x）の事後サンプルを観点ごとに縦長でまとめる。
h1_beta_draws <- function(fits) {
  do.call(rbind, lapply(names(fits), function(a) {
    data.frame(aspect = a, beta = posterior::as_draws_df(fits[[a]])$b_x)
  }))
}

# h1_results() のうち，全体の結果（対数オッズと評定点）を1観点1行にまとめる。
h1_population_table <- function(results) {
  do.call(rbind, lapply(names(results), function(a) {
    r <- results[[a]]
    data.frame(
      aspect = a,
      beta_mean = r$population_log_odds[["mean"]],
      beta_lower95 = r$population_log_odds[["lower95"]],
      beta_upper95 = r$population_log_odds[["upper95"]],
      pr_human_better = r$population_log_odds[["Pr_human_better"]],
      points_mean = r$equal_scale_average_points[["mean"]],
      points_lower95 = r$equal_scale_average_points[["lower95"]],
      points_upper95 = r$equal_scale_average_points[["upper95"]],
      scale_sd_median = r$scale_heterogeneity[["50%"]]
    )
  }))
}

# 尺度別の効果を縦長にまとめる（フォレストプロット用）。
# type = "scale_log_odds"（対数オッズ）または "scale_points"（評定点）。
h1_scale_table <- function(results, type = c("scale_log_odds", "scale_points")) {
  type <- match.arg(type)
  do.call(rbind, lapply(names(results), function(a) {
    m <- results[[a]][[type]]
    data.frame(aspect = a, scale_id = as.integer(rownames(m)),
               mean = m[, "mean"], lower95 = m[, "lower95"],
               upper95 = m[, "upper95"], row.names = NULL)
  }))
}

# ベイズファクター（Savage–Dickey法） -----------------------------------------
# H0: beta = 0 は H1: beta ~ Cauchy(0, r × π/√3) の beta を 0 に固定した入れ子モデルであり，
# beta 以外のパラメータの事前分布は両モデルで同じ（beta とも独立）。このとき
#   BF01 = p(beta = 0 | データ, H1) / p(beta = 0 | H1)
# が成り立つ（Savage–Dickey 密度比）。分母は Cauchy 分布なので解析的に計算し，
# 分子は beta の事後サンプルから 0 における密度を推定する。

# 事後サンプル z の点 at における密度（カーネル密度推定）。
h1_density_at <- function(z, at = 0) {
  dens <- stats::density(z, n = 2^12, from = min(z, at) - 0.5, to = max(z, at) + 0.5)
  stats::approx(dens$x, dens$y, xout = at)$y
}

# 証拠の強さの言葉。bf は「左側のモデル」を支持する BF（例：BF01 なら差なし）。
h1_bf_label <- function(bf, favor, against) {
  ifelse(bf >= 10, paste0(favor, "を強く支持"),
  ifelse(bf >= 3, paste0(favor, "を支持"),
  ifelse(bf > 1 / 3, "判断保留",
  ifelse(bf > 1 / 10, paste0(against, "を支持"), paste0(against, "を強く支持")))))
}

# 1つの観点の主解析モデルから，両側・片側の BF をまとめて計算する。
#   BF10      : 平均差あり（両方向） vs 平均差0
#   BF_plus0  : 人手版が高い（beta > 0，半 Cauchy 事前分布） vs 平均差0
#   BF_minus0 : AI版が高い（beta < 0） vs 平均差0
# 片側 BF は BF10 × Pr(方向 | データ) / Pr(方向 | 事前) で求める（Morey & Wagenmakers, 2014）。
# brms::hypothesis() は事前サンプルの密度推定を使うが，Cauchy は裾が重く 0 での密度が
# 不安定になるので，確認には使わない（分母は解析的に求まる）。
h1_savage_dickey <- function(fit, prior_r = h1_prior_r) {
  prior_b <- fit$prior$prior[fit$prior$class == "b" & fit$prior$coef == ""]
  stopifnot(identical(prior_b, h1_beta_prior(prior_r)))
  draws <- posterior::as_draws_df(fit)
  b <- draws$b_x
  prior0 <- h1_prior_density(0, prior_r)
  post0 <- h1_density_at(b, 0)
  bf01 <- post0 / prior0
  pr_pos <- mean(b > 0)
  # 計算の安定性の確認：(1) 正規近似，(2) チェーンごと
  bf01_normal <- stats::dnorm(0, mean(b), stats::sd(b)) / prior0
  bf01_chain <- tapply(b, draws$.chain, function(z) h1_density_at(z, 0) / prior0)
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
h1_bf_prior_curve <- function(fit, bf10_base, base_r = h1_prior_r,
                              prior_rs = sort(unique(c(seq(0.1, 1.5, by = 0.05),
                                                       h1_prior_r)))) {
  b <- posterior::as_draws_df(fit)$b_x
  do.call(rbind, lapply(prior_rs, function(r) {
    w <- h1_prior_density(b, r) / h1_prior_density(b, base_r)
    bf10 <- bf10_base * mean(w)
    pr_pos <- sum(w[b > 0]) / sum(w)
    data.frame(prior_r = r, BF10 = bf10, BF01 = 1 / bf10,
               BF_plus0 = bf10 * pr_pos / 0.5, BF_minus0 = bf10 * (1 - pr_pos) / 0.5,
               ess_w = sum(w)^2 / sum(w^2))
  }))
}

# ブリッジサンプリングの反復 ----------------------------------------------------
# brms の文書は，ブリッジサンプリングには通常の推定より多くの事後サンプル（目安として
# 10倍程度）が必要で，複数回計算して安定性を確かめるよう勧めている。そこで，
#   (1) 事後サンプルを増やして H1・H0 を別々に推定し（h1_fit_bf），
#   (2) 乱数シードを変えて bridge_sampler() を複数回実行する（h1_bridge_repeat）。
# どちらも h1_cached() で保存する。キャッシュのキーには brmsfit を直接入れず，
# 推定の入力（データ・式・設定）を入れる。
# 主解析と同じく cmdstanr バックエンドで推定し，チェーン内を threading() で並列化する。

h1_fit_bf <- function(d, aspect_name, null_model = FALSE, prior_r = h1_prior_r,
                      iter = 12000, warmup = 2000,
                      threads_per_chain = h1_threads_per_chain(), force_refit = FALSE) {
  stopifnot(requireNamespace("brms", quietly = TRUE),
            requireNamespace("cmdstanr", quietly = TRUE))
  dd <- droplevels(d[d$aspect == aspect_name, ])
  random <- "(1 + x || subject_id) + (1 + x || scale_id) + (1 | pair_id)"
  model_formula <- stats::as.formula(paste(
    if (null_model) "rating ~ 1 +" else "rating ~ x +", random))
  priors <- c(brms::set_prior("normal(0, 2.5)", class = "Intercept"),
              brms::set_prior("normal(0, 1)", class = "sd"))
  if (!null_model) {
    priors <- c(priors, brms::set_prior(h1_beta_prior(prior_r), class = "b"))
  }
  signature <- h1_fit_bf_signature(dd, aspect_name, null_model, prior_r, iter, warmup)
  h1_cached(if (null_model) "bf_null" else "bf_full", signature = signature,
            force_refit = force_refit, compute = function() brms::brm(
    formula = model_formula, data = dd, family = brms::cumulative("logit"),
    prior = priors, sample_prior = "no",
    chains = 4, cores = 4, iter = iter, warmup = warmup, seed = 20261006,
    control = list(adapt_delta = 0.99, max_treedepth = 15),
    backend = "cmdstanr", threads = brms::threading(threads_per_chain),
    save_pars = brms::save_pars(all = TRUE)
  ))
}

# H0 は beta を含まないので事前分布を変えても同じモデルになる。保存済みの H0 の推定結果と
# ブリッジサンプリングの結果を使い続けられるよう，H0 のキーは正規事前分布のときと同じ形
# （slope_prior_sd = NA）のままにし，H1 だけ事前分布の文字列をキーに入れる。
h1_fit_bf_signature <- function(dd, aspect_name, null_model, prior_r, iter, warmup) {
  list(data = dd, aspect = aspect_name, null_model = null_model,
       slope_prior_sd = if (null_model) NA else h1_beta_prior(prior_r),
       iter = iter, warmup = warmup, backend = "cmdstanr")
}

# 1つのモデルについて bridge_sampler() を seeds の数だけ繰り返し，対数周辺尤度を返す。
# bridgesampling の警告（"logml could not be estimated within maxiter" など）も記録する。
h1_bridge_repeat <- function(fit, signature, seeds = 1:5, maxiter = 5000,
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
    h1_cached("bridge", signature = list(fit = signature, seed = seed,
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

# 1つの観点について，事後サンプルを増やしたモデルでブリッジサンプリングを反復し，
# 同じ事後サンプルから求めた Savage–Dickey の BF と並べる。
h1_bridge_check <- function(d, aspect_name, seeds = 1:5, iter = 12000, warmup = 2000,
                            maxiter = 5000) {
  fit1 <- h1_fit_bf(d, aspect_name, null_model = FALSE, iter = iter, warmup = warmup)
  fit0 <- h1_fit_bf(d, aspect_name, null_model = TRUE, iter = iter, warmup = warmup)
  dd <- droplevels(d[d$aspect == aspect_name, ])
  sig1 <- h1_fit_bf_signature(dd, aspect_name, FALSE, h1_prior_r, iter, warmup)
  sig0 <- h1_fit_bf_signature(dd, aspect_name, TRUE, h1_prior_r, iter, warmup)
  b1 <- h1_bridge_repeat(fit1, sig1, seeds = seeds, maxiter = maxiter)
  b0 <- h1_bridge_repeat(fit0, sig0, seeds = seeds, maxiter = maxiter)
  bridge <- data.frame(
    aspect = aspect_name, seed = seeds,
    logml_h1 = b1$logml, logml_h0 = b0$logml,
    BF10 = exp(b1$logml - b0$logml),
    error_h1 = b1$percentage_error, error_h0 = b0$percentage_error,
    n_warnings = b1$n_warnings + b0$n_warnings
  )
  # Savage–Dickey（sample_prior = "no" なので hypothesis() は使わず，密度比だけ求める）
  b <- posterior::as_draws_df(fit1)$b_x
  sd_bf10 <- h1_prior_density(0, h1_prior_r) / h1_density_at(b, 0)
  list(bridge = bridge, BF10_savage_dickey = sd_bf10,
       diagnostics = h1_diag_table(list(H1 = fit1, H0 = fit0)))
}

# 仮説に対する判定 ------------------------------------------------------------
# 判定規則（本文「判定の規則」と対応）
#   方向の予測（人手版 > AI版）: 95%信用区間で判定し，片側 BF（BF+0）で証拠の強さを示す。
#   差なしの予測              : BF01 で判定する。
# sens は h1_bf_prior_curve() の結果（prior_r = h1_sens_rs を含む）。
h1_verdict <- function(aspect, prediction, pop, sd_bf, sens) {
  p <- pop[pop$aspect == aspect, ]
  s <- sd_bf[sd_bf$aspect == aspect, ]
  ss <- sens[sens$aspect == aspect & h1_r_in(sens$prior_r, h1_sens_rs), ]
  if (prediction == "human_better") {
    ci <- if (p$beta_lower95 > 0) "支持（95%CrIが0より上）"
          else if (p$beta_upper95 < 0) "予測と逆（95%CrIが0より下）"
          else "支持されない（95%CrIが0を含む）"
    bf_main <- s$BF_plus0
    evidence <- h1_bf_label(1 / bf_main, "平均差0", "人手版優位")
    labels <- h1_bf_label(1 / ss$BF_plus0, "平均差0", "人手版優位")
    ss_bf <- ss$BF_plus0
    bf_text <- sprintf("BF+0 = %.3g（BF0+ = %.3g）", bf_main, 1 / bf_main)
  } else {
    ci <- sprintf("参考：95%%CrI [%.2f, %.2f]", p$beta_lower95, p$beta_upper95)
    bf_main <- s$BF01
    evidence <- h1_bf_label(bf_main, "平均差0", "平均差あり")
    labels <- h1_bf_label(ss$BF01, "平均差0", "平均差あり")
    ss_bf <- ss$BF01
    bf_text <- sprintf("BF01 = %.3g", bf_main)
  }
  # 頑健性は3区分（≥3／1/3〜3／≤1/3）が Cauchy の尺度 r で変わらないかで判断する
  zone <- cut(ss_bf, c(0, 1 / 3, 3, Inf))
  data.frame(
    aspect = aspect,
    prediction = if (prediction == "human_better") "人手版 > AI版" else "差なし",
    beta = sprintf("%.2f [%.2f, %.2f]", p$beta_mean, p$beta_lower95, p$beta_upper95),
    pr_human_better = sprintf("%.1f%%", 100 * p$pr_human_better),
    ci_verdict = ci,
    bayes_factor = bf_text,
    bf_verdict = evidence,
    prior_robust = if (length(unique(zone)) == 1) "r = 1/2〜1で同じ区分" else
      paste0("r で区分が変わる（", paste(unique(labels), collapse = "／"), "）"),
    points = sprintf("%+.2f [%+.2f, %+.2f]", p$points_mean, p$points_lower95, p$points_upper95)
  )
}
