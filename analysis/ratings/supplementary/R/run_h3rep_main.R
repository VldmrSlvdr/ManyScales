# ============================================================
# 仮説3 独立再現 本計算(Al-khwa で実行する)
#   主分析: 除外基準3なし・21尺度。差得点モデルと評定モデルを H1/H0 で推定し，
#           bridgesampling(warp3 × 10回)で BF01，Savage–Dickey で検算，δ の HDI と ROPE を要約する
#   参考記録: 尺度ごとの BayesFactor::ttestBF(差得点は対応あり，評定は2群)
#   探索パート: r_scale・分散の事前分布の幅・LKJ・回答者の効果・先に答えた版・除外基準3あり・累積プロビット
#   使い方(プロジェクトのルートで):
#     Rscript supplementary/R/run_h3rep_main.R <task> [iter] [warmup]
#     task = main_diff | main_rating | scale_bf |
#            x_rscale_{diff,rating}_{half,one,sqrt2} | x_prior_{half,double}_{diff,rating} |
#            x_lkj | x_person | x_person_normal | x_order | x_excl3_{diff,rating} |
#            x_ordinal_{h1,h0,merge} | collect
#   結果: supplementary/results/h3rep_main/(stanfit 本体は fits/ に置き，手元へは戻さない)
# ============================================================

# ----- パッケージと関数 -----
pacman::p_load(here, tidyverse, rstan, bridgesampling, BayesFactor, bayestestR, logspline, posterior, brms)
source(here::here("supplementary", "R", "hypothesis_3_replication_functions.R"))
# コンパイル済みのモデルを .stan の隣に保存して使い回す
rstan::rstan_options(auto_write = TRUE)

# ----- 引数 -----
args <- commandArgs(trailingOnly = TRUE)
# 何をするか
task <- if (length(args) >= 1) args[1] else "collect"
# 反復の数(本計算は 22000，ウォームアップ 2000 = 事後 4 × 20,000)
iter <- if (length(args) >= 2) as.integer(args[2]) else 22000L
warmup <- if (length(args) >= 3) as.integer(args[3]) else 2000L
# bridgesampling の繰り返しの数
n_rep <- 10L
# 出力先
out_dir <- here::here("supplementary", "results", "h3rep_main")
fit_dir <- file.path(out_dir, "fits")
dir.create(fit_dir, recursive = TRUE, showWarnings = FALSE)
# データの置き場所(主分析 = 除外基準3なし，探索 = 除外基準3あり)
data_dir_main <- here::here("primary", "data", "3_data_processed")
data_dir_excl3 <- here::here("sensitivity", "data", "3_data_processed")
# 探索パートの r_scale(計画案 S1 の値。主分析は √2/2)
rscale_values <- c(half = 1 / 2, one = 1, sqrt2 = sqrt(2))
# 分散の事前分布の幅の倍率(先生の確認B: 半分・倍)
prior_mult <- c(half = 0.5, double = 2)

# ----- 記録の道具 -----
# 経過を記録する(print/cat は使わずログファイルへ追記する)
log_file <- file.path(out_dir, sprintf("log_%s.log", task))
log_line <- function(...) {
  # 時刻を付けて1行追記する
  write(paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), paste0(...)), log_file, append = TRUE)
  # 何も返さない
  return(invisible(NULL))
}
# 経過時間を秒で測る
secs <- function(t0) {
  # 開始時刻からの秒数を返す
  return(as.numeric(difftime(Sys.time(), t0, units = "secs")))
}
# 表を1行の文字列にする(ログ用)
flat <- function(x) {
  # データフレームを文字列にして縦棒でつなぐ
  return(paste(utils::capture.output(as.data.frame(x)), collapse = " | "))
}


# ============================================================
# 共通の道具
# ============================================================

# BF01 の記号(○ = BF01 > 3，× = BF01 < 1/3，△ = その間)
bf_mark <- function(bf01) {
  # 記号を返す
  return(dplyr::case_when(bf01 > 3 ~ "o", bf01 < 1 / 3 ~ "x", TRUE ~ "tri"))
}

# 抽出の行列の列ごとに事後平均値・95% HDI を出す(δ の列には ROPE に入る確率を付ける)
summarise_cols <- function(mat) {
  out <- purrr::map(colnames(mat), function(p) {
    # δ(delta[1] または b_x)にだけ ROPE に入る確率を付ける
    is_delta <- p %in% c("delta[1]", "b_x")
    h3_summarise_draws(as.vector(mat[, p]), rope = if (is_delta) H3_ROPE else NULL) %>%
      dplyr::mutate(par = p, .before = 1)
  }) %>% dplyr::bind_rows()
  # 要約を返す
  return(out)
}

# 全パラメータの収束の診断を要約する(H0 で0に固定した値のように R̂ が NA のものは除く)
conv_summary <- function(stanfit, hyp) {
  # 保存されている全パラメータ
  pars <- stanfit@sim$pars_oi
  conv <- h3_check_convergence(stanfit, pars) %>% dplyr::filter(!is.na(rhat))
  # 最悪値を1行にする
  out <- conv %>%
    dplyr::summarise(n_par = dplyr::n(), max_rhat = max(rhat), min_ess_bulk = min(ess_bulk),
                     min_ess_tail = min(ess_tail), n_divergent = max(n_divergent)) %>%
    dplyr::mutate(hyp = hyp, .before = 1)
  # 要約を返す
  return(out)
}

# Savage–Dickey 比(δ の抽出と Cauchy の尺度から。h3_bf_savage_dickey() と同じ計算を抽出から行う)
sd_from_draws <- function(delta, r_scale) {
  # 事後密度を logspline で滑らかにして0での値を求める
  ls <- logspline::logspline(delta)
  # BF01 = 0 での事後密度 / 0 での事前密度
  out <- logspline::dlogspline(0, ls) / stats::dcauchy(0, 0, r_scale)
  return(out)
}

# 基準(a)〜(c)の値を1行にまとめる
make_criteria <- function(log_bf01, bf01_sd, bf01_normal) {
  # 10回の log BF01 の中央値
  med <- stats::median(log_bf01)
  out <- tibble::tibble(
    bf01 = exp(med),                                        # BF01(10回の中央値)
    bf01_min = exp(min(log_bf01)), bf01_max = exp(max(log_bf01)),
    a_range_log_bf01 = max(log_bf01) - min(log_bf01),       # (a) 最大 − 最小
    a_pass = a_range_log_bf01 < 0.1,
    b_marks = paste(unique(bf_mark(exp(log_bf01))), collapse = ","),  # (b) 記号が10回とも同じか
    b_pass = dplyr::n_distinct(bf_mark(exp(log_bf01))) == 1,
    bf01_sd = bf01_sd,                                      # (c) Savage–Dickey
    c_log_diff = log(bf01_sd) - med,
    # (c) の操作的定義: 記号が同じで，比が 0.8〜1.25 倍の内側(|log 比| < log 1.25)
    c_pass = bf_mark(bf01_sd) == bf_mark(exp(med)) & abs(c_log_diff) < log(1.25),
    bf01_normal = bf01_normal                               # 参考: normal 法の BF01
  )
  return(out)
}


# ============================================================
# Stan のモデル1組(H1 と H0)を推定して BF と要約を出す
# ============================================================
run_pair <- function(tag, model_name, data_h1, data_h0, top_pars, scale_pars = character(0),
                     h0_from = NULL, keep_scale_draws = FALSE) {
  # コンパイル済みのモデル(.stan の隣の rds)を読む
  model <- rstan::stan_model(here::here("supplementary", "stan", sprintf("h3rep_%s.stan", model_name)))
  times <- list()
  # 抽出なしの stanfit(bridgesampling の log_prob 用。保存した stanfit を渡すと止まるため)
  empty_fit <- function(stan_data) {
    sd_in <- stan_data[setdiff(names(stan_data), "scale_ids")]
    out <- rstan::sampling(model, data = sd_in, chains = 0)
    return(out)
  }
  # warp3 を10回(乱数を固定)
  bridge_rep <- function(fit, empty) {
    set.seed(H3_SEED)
    out <- bridgesampling::bridge_sampler(fit, stanfit_model = empty, method = "warp3",
                                          repetitions = n_rep, silent = TRUE)
    return(out)
  }
  # normal 法を1回(パーセント誤差の参考値)
  bridge_normal <- function(fit, empty) {
    set.seed(H3_SEED)
    b <- bridgesampling::bridge_sampler(fit, stanfit_model = empty, method = "normal", silent = TRUE)
    # 誤差の計算が退化して止まっても本体を止めない(参考値なので NA にする)
    pct <- tryCatch(100 * bridgesampling::error_measures(b)$cv, error = function(e) NA_real_)
    out <- tibble::tibble(logml = b$logml, pct_error = pct)
    return(out)
  }
  # stanfit を推定するか，保存済みなら読む
  get_fit <- function(stan_data, hyp) {
    path <- file.path(fit_dir, sprintf("fit_%s_%s.rds", tag, hyp))
    if (file.exists(path)) {
      log_line(hyp, ": loaded saved fit")
      return(readRDS(path))
    }
    t0 <- Sys.time()
    fit <- h3_fit(model, stan_data, iter = iter, warmup = warmup)
    times[[paste0("sampling_", hyp)]] <<- secs(t0)
    log_line(hyp, " sampled in ", round(secs(t0), 1), " s")
    saveRDS(fit, path)
    return(fit)
  }

  # ----- H1 -----
  fit_h1 <- get_fit(data_h1, "h1")
  conv <- conv_summary(fit_h1, "H1")
  # 最上位のパラメータと尺度ごとのパラメータを要約する
  top <- summarise_cols(as.matrix(fit_h1, pars = top_pars)) %>% dplyr::mutate(hyp = "H1", .before = 1)
  scales <- purrr::map(scale_pars, function(p) {
    h3_summarise_scales(fit_h1, p, data_h1$scale_ids) %>% dplyr::mutate(par = p, .before = 1)
  }) %>% dplyr::bind_rows()
  # フォレストプロット用に尺度ごとの抽出を間引いて残す(10本に1本)
  scale_draws <- if (keep_scale_draws) {
    purrr::map(c(scale_pars, intersect(top_pars, c("mu", "beta", "mu_new", "beta_new"))), function(p) {
      m <- as.matrix(fit_h1, pars = p)
      m[seq(1, nrow(m), by = 10), , drop = FALSE]
    }) %>% purrr::set_names(c(scale_pars, intersect(top_pars, c("mu", "beta", "mu_new", "beta_new"))))
  } else NULL
  # bridgesampling(H1)
  empty_h1 <- empty_fit(data_h1)
  t0 <- Sys.time()
  ml1 <- bridge_rep(fit_h1, empty_h1)
  times$bridge_warp3_h1 <- secs(t0)
  normal_h1 <- bridge_normal(fit_h1, empty_h1)
  log_line("H1 bridge warp3 x ", n_rep, " in ", round(times$bridge_warp3_h1, 1), " s")
  # Savage–Dickey
  bf01_sd <- sd_from_draws(as.vector(as.matrix(fit_h1, pars = "delta")), data_h1$r_scale)
  rm(fit_h1)
  gc()

  # ----- H0(別の組から借りるか，推定する) -----
  if (!is.null(h0_from)) {
    # H0 は r_scale に依存しないので，主分析の H0 の bridgesampling の結果を使う
    src_path <- file.path(out_dir, sprintf("pair_%s.rds", h0_from))
    # 主分析が終わるまで待つ(最大 12 時間)
    n_wait <- 0
    while (!file.exists(src_path) && n_wait < 720) {
      Sys.sleep(60)
      n_wait <- n_wait + 1
    }
    src <- readRDS(src_path)
    logml_h0 <- src$bridge$logml_h0
    normal_h0 <- src$normal %>% dplyr::filter(hyp == "H0") %>% dplyr::select(logml, pct_error)
    log_line("H0 bridge results taken from ", h0_from)
  } else {
    fit_h0 <- get_fit(data_h0, "h0")
    conv <- dplyr::bind_rows(conv, conv_summary(fit_h0, "H0"))
    # H0 のパラメータも要約する(0 に固定した δ・μ・β は除く)
    h0_pars <- setdiff(top_pars, c("delta", "mu", "beta", "mean_ai", "mean_human"))
    top <- dplyr::bind_rows(top, summarise_cols(as.matrix(fit_h0, pars = h0_pars)) %>%
                              dplyr::mutate(hyp = "H0", .before = 1))
    empty_h0 <- empty_fit(data_h0)
    t0 <- Sys.time()
    ml0 <- bridge_rep(fit_h0, empty_h0)
    times$bridge_warp3_h0 <- secs(t0)
    normal_h0 <- bridge_normal(fit_h0, empty_h0)
    log_line("H0 bridge warp3 x ", n_rep, " in ", round(times$bridge_warp3_h0, 1), " s")
    logml_h0 <- ml0$logml
    rm(fit_h0)
    gc()
  }

  # ----- BF と基準 -----
  bridge <- tibble::tibble(rep = seq_len(n_rep), logml_h1 = ml1$logml, logml_h0 = logml_h0) %>%
    dplyr::mutate(log_bf01 = logml_h0 - logml_h1, bf01 = exp(log_bf01), mark = bf_mark(bf01))
  normal <- dplyr::bind_rows(normal_h1 %>% dplyr::mutate(hyp = "H1", .before = 1),
                             normal_h0 %>% dplyr::mutate(hyp = "H0", .before = 1))
  crit <- make_criteria(bridge$log_bf01, bf01_sd, exp(normal_h0$logml - normal_h1$logml)) %>%
    dplyr::mutate(tag = tag, model = model_name, r_scale = data_h1$r_scale,
                  s_tau = data_h1$s_tau, s_sigma = data_h1$s_sigma, .before = 1)
  log_line("convergence: ", flat(conv))
  log_line("criteria: ", flat(crit))
  # 保存する(stanfit は含めない)
  res <- list(tag = tag, model = model_name, iter = iter, warmup = warmup,
              times = tibble::as_tibble(times), conv = conv, top = top, scales = scales,
              scale_draws = scale_draws, bridge = bridge, normal = normal, criteria = crit,
              n_obs = data_h1$N, n_scales = data_h1$J, scale_ids = data_h1$scale_ids,
              session = utils::sessionInfo())
  saveRDS(res, file.path(out_dir, sprintf("pair_%s.rds", tag)))
  return(res)
}


# ============================================================
# 累積プロビット(brms，rstan バックエンド)
#   重いので H1 と H0 を別のプロセスで並行に回す(run_ordinal_half)。
#   bridgesampling は推定と同じセッションで行い(保存した fit を別セッションで渡すと止まるため)，
#   両方がそろってから run_ordinal_merge で BF と基準をまとめる
# ============================================================
run_ordinal_half <- function(tag, wording_long, hyp) {
  times <- list()
  fz <- if (hyp == "h1") 0 else 1
  # 推定する
  t0 <- Sys.time()
  fit <- h3_fit_ordinal(wording_long, fix_zero = fz, iter = iter, warmup = warmup)
  times[[paste0("sampling_", hyp)]] <- secs(t0)
  log_line("ordinal ", hyp, " sampled in ", round(secs(t0), 1), " s")
  saveRDS(fit, file.path(fit_dir, sprintf("fit_%s_%s.rds", tag, hyp)))
  # 収束
  conv <- conv_summary(fit$fit, toupper(hyp))
  log_line("convergence: ", flat(conv))
  # 要約(版の効果 b_x = 潜在尺度の標準化差 = δ，尺度間 SD)
  vars <- c(if (hyp == "h1") "b_x", "sd_scale_f__Intercept", "sd_scale_f__x")
  top <- summarise_cols(as.matrix(fit, variable = vars)) %>% dplyr::mutate(hyp = toupper(hyp), .before = 1)
  # bridgesampling(warp3 × 10，乱数を固定)
  t0 <- Sys.time()
  set.seed(H3_SEED)
  ml <- brms::bridge_sampler(fit, method = "warp3", repetitions = n_rep, silent = TRUE)
  times[[paste0("bridge_warp3_", hyp)]] <- secs(t0)
  log_line("ordinal ", hyp, " bridge warp3 in ", round(secs(t0), 1), " s")
  # normal 法を1回(参考値)
  set.seed(H3_SEED)
  b <- brms::bridge_sampler(fit, method = "normal", silent = TRUE)
  # 誤差の計算が退化して止まっても本体を止めない(参考値なので NA にする)
  pct <- tryCatch(100 * bridgesampling::error_measures(b)$cv, error = function(e) NA_real_)
  normal <- tibble::tibble(hyp = toupper(hyp), logml = b$logml, pct_error = pct)
  # Savage–Dickey(H1 のみ)
  bf01_sd <- if (hyp == "h1") sd_from_draws(as.vector(as.matrix(fit, variable = "b_x")), H3_R_SCALE) else NA_real_
  res <- list(hyp = hyp, times = tibble::as_tibble(times), conv = conv, top = top,
              logml = ml$logml, normal = normal, bf01_sd = bf01_sd,
              n_obs = nrow(wording_long), n_scales = dplyr::n_distinct(wording_long$scale_id),
              session = utils::sessionInfo())
  saveRDS(res, file.path(out_dir, sprintf("half_%s_%s.rds", tag, hyp)))
  return(res)
}

run_ordinal_merge <- function(tag) {
  # 両方の結果がそろうまで待つ(最大 12 時間)
  paths <- file.path(out_dir, sprintf("half_%s_%s.rds", tag, c("h1", "h0")))
  n_wait <- 0
  while (!all(file.exists(paths)) && n_wait < 720) {
    Sys.sleep(60)
    n_wait <- n_wait + 1
  }
  h1 <- readRDS(paths[1])
  h0 <- readRDS(paths[2])
  normal <- dplyr::bind_rows(h1$normal, h0$normal)
  bridge <- tibble::tibble(rep = seq_len(n_rep), logml_h1 = h1$logml, logml_h0 = h0$logml) %>%
    dplyr::mutate(log_bf01 = logml_h0 - logml_h1, bf01 = exp(log_bf01), mark = bf_mark(bf01))
  crit <- make_criteria(bridge$log_bf01, h1$bf01_sd, exp(h0$normal$logml - h1$normal$logml)) %>%
    dplyr::mutate(tag = tag, model = "ordinal", r_scale = H3_R_SCALE, s_tau = 0.5, s_sigma = NA_real_,
                  .before = 1)
  log_line("criteria: ", flat(crit))
  res <- list(tag = tag, model = "ordinal", iter = iter, warmup = warmup,
              times = dplyr::bind_cols(h1$times, h0$times), conv = dplyr::bind_rows(h1$conv, h0$conv),
              top = dplyr::bind_rows(h1$top, h0$top), scales = tibble::tibble(),
              scale_draws = NULL, bridge = bridge, normal = normal, criteria = crit,
              n_obs = h1$n_obs, n_scales = h1$n_scales, session = utils::sessionInfo())
  saveRDS(res, file.path(out_dir, sprintf("pair_%s.rds", tag)))
  return(res)
}


# ============================================================
# 課題ごとの実行
# ============================================================
# 最上位の要約に使うパラメータ
top_diff <- c("delta", "mu", "tau", "sigma", "mu_new")
top_rating <- c("delta", "beta", "alpha", "tau_alpha", "tau_beta", "sigma", "beta_new", "mean_ai", "mean_human")
top_lkj <- c("delta", "beta", "alpha", "tau_ab", "rho_ab", "sigma", "beta_new")

if (task != "collect") {
  t_task <- Sys.time()
  log_line("start task ", task, " (iter = ", iter, ", warmup = ", warmup, ")")
  # ----- データ -----
  data_dir <- if (grepl("excl3", task)) data_dir_excl3 else data_dir_main
  wording_long <- h3_read_wording(data_dir)
  diff_df <- h3_make_diff(wording_long)
  log_line("data (", data_dir, "): wording rows = ", nrow(wording_long), ", diff rows = ", nrow(diff_df),
           ", scales = ", dplyr::n_distinct(wording_long$scale_id),
           ", persons = ", dplyr::n_distinct(wording_long$subject_id))

  # モデルの種類に合わせて H1/H0 のデータを作る
  pair_data <- function(model_name, ...) {
    out <- if (model_name == "diff") {
      list(h1 = h3_stan_data_diff(diff_df, fix_zero = 0, ...), h0 = h3_stan_data_diff(diff_df, fix_zero = 1, ...))
    } else {
      list(h1 = h3_stan_data_rating(wording_long, fix_zero = 0, ...),
           h0 = h3_stan_data_rating(wording_long, fix_zero = 1, ...))
    }
    return(out)
  }

  if (task %in% c("main_diff", "main_rating", "x_excl3_diff", "x_excl3_rating")) {
    # ----- 主分析と除外基準3あり(同じ手順) -----
    model_name <- sub("^(main|x_excl3)_", "", task)
    tag <- task
    pd <- pair_data(model_name)
    run_pair(tag, model_name, pd$h1, pd$h0,
             top_pars = if (model_name == "diff") top_diff else top_rating,
             scale_pars = if (model_name == "diff") "mu_scale" else c("beta_scale", "alpha_scale"),
             keep_scale_draws = grepl("^main", task))
  } else if (grepl("^x_rscale_", task)) {
    # ----- r_scale を変える(H1 だけ推定し，H0 は主分析のものを使う) -----
    parts <- strsplit(sub("^x_rscale_", "", task), "_")[[1]]
    model_name <- parts[1]
    pd <- pair_data(model_name, r_scale = rscale_values[[parts[2]]])
    run_pair(task, model_name, pd$h1, pd$h0,
             top_pars = if (model_name == "diff") top_diff else top_rating,
             h0_from = paste0("main_", model_name))
  } else if (grepl("^x_prior_", task)) {
    # ----- 分散の事前分布の幅を半分・倍(尺度間 SD と残差 SD の両方) -----
    parts <- strsplit(sub("^x_prior_", "", task), "_")[[1]]
    k <- prior_mult[[parts[1]]]
    model_name <- parts[2]
    pd <- pair_data(model_name, s_tau = 0.5 * k, s_sigma = 1 * k)
    run_pair(task, model_name, pd$h1, pd$h0, top_pars = if (model_name == "diff") top_diff else top_rating)
  } else if (task == "x_lkj") {
    # ----- 評定モデルに α_j と β_j の相関(LKJ η = 1) -----
    pd <- pair_data("rating", lkj_eta = 1)
    run_pair(task, "rating_lkj", pd$h1, pd$h0, top_pars = top_lkj)
  } else if (task == "x_person") {
    # ----- 差得点モデルに回答者の効果 -----
    pd <- pair_data("diff", use_person = 1)
    run_pair(task, "diff", pd$h1, pd$h0, top_pars = c(top_diff, "tau_person", "sd_unit"))
  } else if (task == "x_order") {
    # ----- 差得点モデルに先に答えた版の項 ＋ 記述統計 -----
    saveRDS(h3_order_describe(diff_df), file.path(out_dir, "order_describe.rds"))
    pd <- pair_data("diff", use_order = 1)
    run_pair(task, "diff", pd$h1, pd$h0, top_pars = c(top_diff, "gamma"))
  } else if (task == "x_person_normal") {
    # ----- 回答者の効果ありで warp3 が不安定だったときの確認: normal 法を10回繰り返す -----
    # (パラメータが約1,880次元あり，warp3 の10回の BF01 がばらついたため。参考値として残す)
    model <- rstan::stan_model(here::here("supplementary", "stan", "h3rep_diff.stan"))
    pd <- pair_data("diff", use_person = 1)
    ml <- purrr::imap(list(h1 = pd$h1, h0 = pd$h0), function(sdat, hyp) {
      # 保存した stanfit と，抽出なしの stanfit(log_prob 用)
      fit <- readRDS(file.path(fit_dir, sprintf("fit_x_person_%s.rds", hyp)))
      empty <- rstan::sampling(model, data = sdat[setdiff(names(sdat), "scale_ids")], chains = 0)
      set.seed(H3_SEED)
      b <- bridgesampling::bridge_sampler(fit, stanfit_model = empty, method = "normal",
                                          repetitions = n_rep, silent = TRUE)
      return(b$logml)
    })
    out <- tibble::tibble(rep = seq_len(n_rep), logml_h1 = ml$h1, logml_h0 = ml$h0) %>%
      dplyr::mutate(log_bf01 = logml_h0 - logml_h1, bf01 = exp(log_bf01), mark = bf_mark(bf01))
    log_line("normal x ", n_rep, ": ", flat(out))
    saveRDS(out, file.path(out_dir, "x_person_normal_reps.rds"))
    readr::write_csv(out, file.path(out_dir, "x_person_normal_reps.csv"))
  } else if (task %in% c("x_ordinal_h1", "x_ordinal_h0")) {
    # ----- 累積プロビット(iter を引数で変えられる。重いので本計算は 12000 で回す) -----
    run_ordinal_half(sprintf("x_ordinal_iter%d", iter), wording_long, sub("^x_ordinal_", "", task))
  } else if (task == "x_ordinal_merge") {
    # ----- 累積プロビットの H1 と H0 をまとめる -----
    run_ordinal_merge(sprintf("x_ordinal_iter%d", iter))
  } else if (task == "scale_bf") {
    # ----- 尺度ごとの参考記録(21本)。r_scale の探索値も同時に出す -----
    rs_all <- c(main = H3_R_SCALE, rscale_values)
    res <- purrr::imap(rs_all, function(r, nm) {
      dplyr::bind_rows(
        h3_scale_bf(wording_long, "diff", r_scale = r) %>% dplyr::mutate(model = "diff", .before = 1),
        h3_scale_bf(wording_long, "rating", r_scale = r) %>% dplyr::mutate(model = "rating", .before = 1)
      ) %>% dplyr::mutate(r_label = nm, r_scale = r, .before = 1)
    }) %>% dplyr::bind_rows()
    saveRDS(res, file.path(out_dir, "scale_bf.rds"))
    # 記述統計(表1)も残す
    saveRDS(h3_describe(wording_long), file.path(out_dir, "describe.rds"))
  }
  log_line("done task ", task, " in ", round(secs(t_task), 1), " s")
}


# ============================================================
# 集計(要約の rds と csv を作る)
# ============================================================
if (task == "collect") {
  # pair_*.rds を全部読む
  files <- list.files(out_dir, pattern = "^pair_.*\\.rds$", full.names = TRUE)
  pairs <- purrr::map(files, readRDS) %>% purrr::set_names(sub("^pair_(.*)\\.rds$", "\\1", basename(files)))
  # δ の行(差得点・評定は delta[1]，累積プロビットは b_x)
  delta_row <- function(p) {
    out <- p$top %>% dplyr::filter(hyp == "H1", par %in% c("delta[1]", "b_x")) %>%
      dplyr::select(delta_mean = mean, delta_hdi_low = hdi_low, delta_hdi_high = hdi_high, p_rope)
    return(out)
  }
  # 評定の単位の差の行(差得点 = mu，評定 = beta)
  raw_row <- function(p) {
    out <- p$top %>% dplyr::filter(hyp == "H1", par %in% c("mu", "beta")) %>%
      dplyr::select(raw_mean = mean, raw_hdi_low = hdi_low, raw_hdi_high = hdi_high)
    if (nrow(out) == 0) out <- tibble::tibble(raw_mean = NA_real_, raw_hdi_low = NA_real_, raw_hdi_high = NA_real_)
    return(out)
  }
  # 収束の最悪値(H1 と H0 を通して)
  conv_row <- function(p) {
    out <- p$conv %>% dplyr::summarise(max_rhat = max(max_rhat), min_ess_bulk = min(min_ess_bulk),
                                       min_ess_tail = min(min_ess_tail), n_divergent = max(n_divergent))
    return(out)
  }
  summary_tbl <- purrr::imap(pairs, function(p, nm) {
    dplyr::bind_cols(p$criteria, delta_row(p), raw_row(p), conv_row(p),
                     tibble::tibble(n_obs = p$n_obs, iter = p$iter, warmup = p$warmup,
                                    pct_normal_h1 = p$normal$pct_error[p$normal$hyp == "H1"],
                                    pct_normal_h0 = p$normal$pct_error[p$normal$hyp == "H0"]))
  }) %>% dplyr::bind_rows() %>%
    # HDI と ROPE の関係(すっぽり含む・一部含む・含まない)と収束の合否
    dplyr::mutate(
      bf_mark = bf_mark(bf01),
      hdi_rope = dplyr::case_when(delta_hdi_low > H3_ROPE[1] & delta_hdi_high < H3_ROPE[2] ~ "inside",
                                  delta_hdi_high < H3_ROPE[1] | delta_hdi_low > H3_ROPE[2] ~ "outside",
                                  TRUE ~ "overlap"),
      conv_pass = max_rhat < 1.01 & min_ess_bulk > 1000 & min_ess_tail > 1000 & n_divergent == 0
    )
  readr::write_csv(summary_tbl, file.path(out_dir, "summary_bf_delta.csv"))
  # 最上位パラメータの全要約
  top_all <- purrr::imap(pairs, function(p, nm) dplyr::mutate(p$top, tag = nm, .before = 1)) %>% dplyr::bind_rows()
  readr::write_csv(top_all, file.path(out_dir, "top_parameters.csv"))
  # 尺度ごとの事後(主分析)
  scales_all <- purrr::imap(pairs[grepl("^main", names(pairs))], function(p, nm) {
    dplyr::mutate(p$scales, tag = nm, .before = 1)
  }) %>% dplyr::bind_rows()
  readr::write_csv(scales_all, file.path(out_dir, "main_scale_posteriors.csv"))
  # 10回の bridgesampling の値
  bridge_all <- purrr::imap(pairs, function(p, nm) dplyr::mutate(p$bridge, tag = nm, .before = 1)) %>% dplyr::bind_rows()
  readr::write_csv(bridge_all, file.path(out_dir, "bridge_reps.csv"))
  # 収束の詳細
  conv_all <- purrr::imap(pairs, function(p, nm) dplyr::mutate(p$conv, tag = nm, .before = 1)) %>% dplyr::bind_rows()
  readr::write_csv(conv_all, file.path(out_dir, "convergence.csv"))
  # 所要時間
  times_all <- purrr::imap(pairs, function(p, nm) dplyr::mutate(p$times, tag = nm, .before = 1)) %>% dplyr::bind_rows()
  readr::write_csv(times_all, file.path(out_dir, "times.csv"))
  # 尺度ごとの参考記録
  if (file.exists(file.path(out_dir, "scale_bf.rds"))) {
    sbf <- readRDS(file.path(out_dir, "scale_bf.rds")) %>%
      dplyr::mutate(bf_mark = bf_mark(bf01),
                    hdi_rope = dplyr::case_when(delta_hdi_low > H3_ROPE[1] & delta_hdi_high < H3_ROPE[2] ~ "inside",
                                                delta_hdi_high < H3_ROPE[1] | delta_hdi_low > H3_ROPE[2] ~ "outside",
                                                TRUE ~ "overlap"))
    readr::write_csv(sbf, file.path(out_dir, "scale_bf.csv"))
  }
  if (file.exists(file.path(out_dir, "order_describe.rds"))) {
    readr::write_csv(readRDS(file.path(out_dir, "order_describe.rds")), file.path(out_dir, "order_describe.csv"))
  }
  if (file.exists(file.path(out_dir, "describe.rds"))) {
    readr::write_csv(readRDS(file.path(out_dir, "describe.rds")), file.path(out_dir, "describe.csv"))
  }
  # 全体をひとつの rds にまとめる(stanfit は含めない)
  saveRDS(list(summary = summary_tbl, top = top_all, scales = scales_all, bridge = bridge_all,
               conv = conv_all, times = times_all,
               scale_draws = purrr::map(pairs[grepl("^main", names(pairs))], "scale_draws")),
          file.path(out_dir, "h3rep_main_results.rds"))
}
