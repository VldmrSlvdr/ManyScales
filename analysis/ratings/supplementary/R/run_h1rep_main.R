# ============================================================
# 仮説1 独立再現 本計算(Al-khwa で実行する)
#   主分析: 観点ごと(意味的忠実性・表現の自然さ・文化的妥当性)に独立の3本。21尺度(TIVS を除く)。
#           差得点モデルと評定モデルを H1/H0 で推定し，bridgesampling(warp3 × 10回)で BF01，
#           Savage–Dickey で検算，δ の HDI と ROPE を要約する。差は AI版 − 人手版
#           文化的妥当性は記録された値のまま使う(Qualtrics で値が逆向きに割り当て済みで，大きいほど文化的に妥当。
#           10/07 の計算では 6 − 値 に逆転していた＝二重の逆転。2026-10-31 に訂正)
#   参考記録: 尺度ごとの BayesFactor::ttestBF(差得点は対応あり，評定は2群)
#   探索パート: r_scale・分散の事前分布の幅・LKJ・先に評価した版・累積プロビット・片側 BF・
#               回答者の効果(推定と Savage–Dickey のみ)・文化的妥当性を 6 − 値 に逆転した版(10/07 の扱い)
#   使い方(プロジェクトのルートで):
#     Rscript supplementary/R/run_h1rep_main.R data                         # 除外・照合・読み方の点検・記述統計
#     Rscript supplementary/R/run_h1rep_main.R <aspect> <task> [iter] [warmup]
#       aspect = sem | nat | cv
#       task   = main_diff | main_rating | scale_bf |
#                x_rscale_{diff,rating}_{half,one,sqrt2} | x_prior_{half,double}_{diff,rating} |
#                x_lkj | x_order | x_person | x_ordinal_{h1,h0,merge} |
#                x_cvrev_{diff,rating}(cv のみ)
#     Rscript supplementary/R/run_h1rep_main.R collect
#   結果: supplementary/results/h1rep_main/(stanfit 本体は fits/ に置き，手元へは戻さない)
# ============================================================

# ----- パッケージと関数 -----
pacman::p_load(here, tidyverse, rstan, bridgesampling, BayesFactor, bayestestR, logspline, posterior, brms)
source(here::here("supplementary", "R", "hypothesis_1_replication_functions.R"))
# コンパイル済みのモデルを .stan の隣に保存して使い回す(仮説3と同じ Stan コード)
rstan::rstan_options(auto_write = TRUE)

# ----- 引数 -----
args <- commandArgs(trailingOnly = TRUE)
# 観点(data と collect は観点なし)
aspect_short <- if (length(args) >= 1) args[1] else "collect"
task <- if (aspect_short %in% c("data", "collect")) aspect_short else args[2]
# 反復の数(本計算は 22000，ウォームアップ 2000 = 事後 4 × 20,000)
arg_iter <- if (aspect_short %in% c("data", "collect")) 2 else 3
iter <- if (length(args) >= arg_iter) as.integer(args[arg_iter]) else 22000L
warmup <- if (length(args) >= arg_iter + 1) as.integer(args[arg_iter + 1]) else 2000L
# bridgesampling の繰り返しの数
n_rep <- 10L
# 出力先(試験は環境変数で別の場所へ)
out_dir <- Sys.getenv("H1REP_OUT", here::here("supplementary", "results", "h1rep_main"))
fit_dir <- file.path(out_dir, "fits")
dir.create(fit_dir, recursive = TRUE, showWarnings = FALSE)
# データの置き場所(専門家データは除外基準3の有無で同一。除外基準3なしの側を使う)
data_root <- here::here("primary", "data")
flags_file <- file.path(data_root, "1_data_flags", "ManyScales_専門_analysis_flags.tsv")
rating_long_file <- file.path(data_root, "3_data_processed", "ManyScales_専門家_rating_long.csv")
# 観点の略号と列の値の対応
aspect_map <- c(sem = "semantic", nat = "naturalness", cv = "cultural_validity")
# 探索パートの r_scale(仮説3と同じ値。主分析は √2/2)
rscale_values <- c(half = 1 / 2, one = 1, sqrt2 = sqrt(2))
# 分散の事前分布の幅の倍率(半分・倍)
prior_mult <- c(half = 0.5, double = 2)
# 課題の名前(ログと結果のファイル名に使う)
tag <- if (aspect_short %in% c("data", "collect")) task else paste0(aspect_short, "_", task)

# ----- 記録の道具 -----
# 経過を記録する(print/cat は使わずログファイルへ追記する)
log_file <- file.path(out_dir, sprintf("log_%s.log", tag))
log_line <- function(...) {
  # 時刻を付けて1行追記する
  write(paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), paste0(...)), log_file, append = TRUE)
  return(invisible(NULL))
}
# 経過時間を秒で測る
secs <- function(t0) {
  return(as.numeric(difftime(Sys.time(), t0, units = "secs")))
}
# 表を1行の文字列にする(ログ用)
flat <- function(x) {
  return(paste(utils::capture.output(as.data.frame(x)), collapse = " | "))
}


# ============================================================
# 共通の道具(仮説3の本計算と同じ)
# ============================================================

# BF01 の記号(o = BF01 > 3，x = BF01 < 1/3，tri = その間)
bf_mark <- function(bf01) {
  return(dplyr::case_when(is.na(bf01) ~ NA_character_, bf01 > 3 ~ "o", bf01 < 1 / 3 ~ "x", TRUE ~ "tri"))
}

# 抽出の行列の列ごとに事後平均値・95% HDI を出す(δ の列には ROPE に入る確率を付ける)
summarise_cols <- function(mat) {
  out <- purrr::map(colnames(mat), function(p) {
    is_delta <- p %in% c("delta[1]", "b_x")
    h3_summarise_draws(as.vector(mat[, p]), rope = if (is_delta) H3_ROPE else NULL) %>%
      dplyr::mutate(par = p, .before = 1)
  }) %>% dplyr::bind_rows()
  return(out)
}

# 全パラメータの収束の診断を要約する(R̂ が NA のものは除く)
conv_summary <- function(stanfit, hyp) {
  pars <- stanfit@sim$pars_oi
  conv <- h3_check_convergence(stanfit, pars) %>% dplyr::filter(!is.na(rhat))
  out <- conv %>%
    dplyr::summarise(n_par = dplyr::n(), max_rhat = max(rhat), min_ess_bulk = min(ess_bulk),
                     min_ess_tail = min(ess_tail), n_divergent = max(n_divergent)) %>%
    dplyr::mutate(hyp = hyp, .before = 1)
  return(out)
}

# Savage–Dickey 比(δ の抽出と Cauchy の尺度から)
sd_from_draws <- function(delta, r_scale) {
  ls <- logspline::logspline(delta)
  out <- logspline::dlogspline(0, ls) / stats::dcauchy(0, 0, r_scale)
  return(out)
}

# 基準(a)〜(c)の値と片側 BF を1行にまとめる
#   片側 BF(予想の向き δ < 0 = 人手版 > AI版): BF0- = BF01 × P(δ<0 | H1 の事前) / P(δ<0 | データ, H1)
#   事前の P(δ<0) は Cauchy が0について対称なので 0.5
make_criteria <- function(log_bf01, bf01_sd, bf01_normal, p_neg) {
  med <- if (all(is.na(log_bf01))) NA_real_ else stats::median(log_bf01)
  out <- tibble::tibble(
    bf01 = exp(med),
    bf01_min = exp(suppressWarnings(min(log_bf01))), bf01_max = exp(suppressWarnings(max(log_bf01))),
    a_range_log_bf01 = suppressWarnings(max(log_bf01) - min(log_bf01)),
    a_pass = a_range_log_bf01 < 0.1,
    b_marks = paste(unique(bf_mark(exp(log_bf01))), collapse = ","),
    b_pass = dplyr::n_distinct(bf_mark(exp(log_bf01))) == 1 & !any(is.na(log_bf01)),
    bf01_sd = bf01_sd,
    c_log_diff = log(bf01_sd) - med,
    # (c) 記号が同じで，比が 0.8〜1.25 倍の内側(|log 比| < log 1.25)。仮説3と同じ操作的定義
    c_pass = bf_mark(bf01_sd) == bf_mark(exp(med)) & abs(c_log_diff) < log(1.25),
    bf01_normal = bf01_normal,
    p_neg = p_neg,
    bf0neg = exp(med) * 0.5 / p_neg,          # 片側 BF0-(bridgesampling の BF01 から)
    bf0neg_sd = bf01_sd * 0.5 / p_neg         # 片側 BF0-(Savage–Dickey から)
  )
  return(out)
}


# ============================================================
# Stan のモデル1組(H1 と H0)を推定して BF と要約を出す
#   do_bridge = FALSE のときは H1 だけを推定し，Savage–Dickey と要約だけを出す(回答者の効果ありのモデル)
# ============================================================
run_pair <- function(tag, model_name, data_h1, data_h0, top_pars, scale_pars = character(0),
                     h0_from = NULL, keep_scale_draws = FALSE, do_bridge = TRUE) {
  model <- rstan::stan_model(here::here("supplementary", "stan", sprintf("h3rep_%s.stan", model_name)))
  times <- list()
  # 抽出なしの stanfit(bridgesampling の log_prob 用)
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
  top <- summarise_cols(as.matrix(fit_h1, pars = top_pars)) %>% dplyr::mutate(hyp = "H1", .before = 1)
  scales <- purrr::map(scale_pars, function(p) {
    h3_summarise_scales(fit_h1, p, data_h1$scale_ids) %>% dplyr::mutate(par = p, .before = 1)
  }) %>% dplyr::bind_rows()
  # フォレストプロット用に尺度ごとの抽出を間引いて残す(10本に1本)
  draw_pars <- c(scale_pars, intersect(top_pars, c("mu", "beta", "mu_new", "beta_new")))
  scale_draws <- if (keep_scale_draws) {
    purrr::map(draw_pars, function(p) {
      m <- as.matrix(fit_h1, pars = p)
      m[seq(1, nrow(m), by = 10), , drop = FALSE]
    }) %>% purrr::set_names(draw_pars)
  } else NULL
  # δ の抽出(Savage–Dickey と片側 BF 用)
  delta_draws <- as.vector(as.matrix(fit_h1, pars = "delta"))
  bf01_sd <- sd_from_draws(delta_draws, data_h1$r_scale)
  p_neg <- mean(delta_draws < 0)

  if (!do_bridge) {
    # ----- bridgesampling をしない(推定と Savage–Dickey のみ) -----
    rm(fit_h1)
    gc()
    bridge <- tibble::tibble(rep = seq_len(n_rep), logml_h1 = NA_real_, logml_h0 = NA_real_,
                             log_bf01 = NA_real_, bf01 = NA_real_, mark = NA_character_)
    normal <- tibble::tibble(hyp = c("H1", "H0"), logml = NA_real_, pct_error = NA_real_)
    crit <- make_criteria(rep(NA_real_, n_rep), bf01_sd, NA_real_, p_neg)
  } else {
    # bridgesampling(H1)
    empty_h1 <- empty_fit(data_h1)
    t0 <- Sys.time()
    ml1 <- bridge_rep(fit_h1, empty_h1)
    times$bridge_warp3_h1 <- secs(t0)
    normal_h1 <- bridge_normal(fit_h1, empty_h1)
    log_line("H1 bridge warp3 x ", n_rep, " in ", round(times$bridge_warp3_h1, 1), " s")
    rm(fit_h1)
    gc()

    # ----- H0(主分析の結果を借りるか，推定する) -----
    if (!is.null(h0_from)) {
      # H0 は r_scale に依存しないので，同じ観点の主分析の H0 の bridgesampling の結果を使う
      src_path <- file.path(out_dir, sprintf("pair_%s.rds", h0_from))
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
    bridge <- tibble::tibble(rep = seq_len(n_rep), logml_h1 = ml1$logml, logml_h0 = logml_h0) %>%
      dplyr::mutate(log_bf01 = logml_h0 - logml_h1, bf01 = exp(log_bf01), mark = bf_mark(bf01))
    normal <- dplyr::bind_rows(normal_h1 %>% dplyr::mutate(hyp = "H1", .before = 1),
                               normal_h0 %>% dplyr::mutate(hyp = "H0", .before = 1))
    crit <- make_criteria(bridge$log_bf01, bf01_sd, exp(normal_h0$logml - normal_h1$logml), p_neg)
  }

  crit <- crit %>%
    dplyr::mutate(tag = tag, model = model_name, r_scale = data_h1$r_scale,
                  s_tau = data_h1$s_tau, s_sigma = data_h1$s_sigma, .before = 1)
  log_line("convergence: ", flat(conv))
  log_line("criteria: ", flat(crit))
  res <- list(tag = tag, model = model_name, iter = iter, warmup = warmup, do_bridge = do_bridge,
              times = tibble::as_tibble(times), conv = conv, top = top, scales = scales,
              scale_draws = scale_draws, bridge = bridge, normal = normal, criteria = crit,
              n_obs = data_h1$N, n_scales = data_h1$J, scale_ids = data_h1$scale_ids,
              session = utils::sessionInfo())
  saveRDS(res, file.path(out_dir, sprintf("pair_%s.rds", tag)))
  return(res)
}


# ============================================================
# 累積プロビット(brms，rstan バックエンド)。H1 と H0 を別のプロセスで回し，merge でまとめる
# ============================================================
run_ordinal_half <- function(tag, wording_long, hyp) {
  times <- list()
  fz <- if (hyp == "h1") 0 else 1
  t0 <- Sys.time()
  fit <- h3_fit_ordinal(wording_long, fix_zero = fz, iter = iter, warmup = warmup)
  times[[paste0("sampling_", hyp)]] <- secs(t0)
  log_line("ordinal ", hyp, " sampled in ", round(secs(t0), 1), " s")
  saveRDS(fit, file.path(fit_dir, sprintf("fit_%s_%s.rds", tag, hyp)))
  conv <- conv_summary(fit$fit, toupper(hyp))
  log_line("convergence: ", flat(conv))
  vars <- c(if (hyp == "h1") "b_x", "sd_scale_f__Intercept", "sd_scale_f__x")
  top <- summarise_cols(as.matrix(fit, variable = vars)) %>% dplyr::mutate(hyp = toupper(hyp), .before = 1)
  t0 <- Sys.time()
  set.seed(H3_SEED)
  ml <- brms::bridge_sampler(fit, method = "warp3", repetitions = n_rep, silent = TRUE)
  times[[paste0("bridge_warp3_", hyp)]] <- secs(t0)
  log_line("ordinal ", hyp, " bridge warp3 in ", round(secs(t0), 1), " s")
  set.seed(H3_SEED)
  b <- brms::bridge_sampler(fit, method = "normal", silent = TRUE)
  pct <- tryCatch(100 * bridgesampling::error_measures(b)$cv, error = function(e) NA_real_)
  normal <- tibble::tibble(hyp = toupper(hyp), logml = b$logml, pct_error = pct)
  bx <- if (hyp == "h1") as.vector(as.matrix(fit, variable = "b_x")) else NA_real_
  bf01_sd <- if (hyp == "h1") sd_from_draws(bx, H3_R_SCALE) else NA_real_
  p_neg <- if (hyp == "h1") mean(bx < 0) else NA_real_
  res <- list(hyp = hyp, times = tibble::as_tibble(times), conv = conv, top = top,
              logml = ml$logml, normal = normal, bf01_sd = bf01_sd, p_neg = p_neg,
              n_obs = nrow(wording_long), n_scales = dplyr::n_distinct(wording_long$scale_id),
              session = utils::sessionInfo())
  saveRDS(res, file.path(out_dir, sprintf("half_%s_%s.rds", tag, hyp)))
  return(res)
}

run_ordinal_merge <- function(tag) {
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
  crit <- make_criteria(bridge$log_bf01, h1$bf01_sd, exp(h0$normal$logml - h1$normal$logml), h1$p_neg) %>%
    dplyr::mutate(tag = tag, model = "ordinal", r_scale = H3_R_SCALE, s_tau = 0.5, s_sigma = NA_real_,
                  .before = 1)
  log_line("criteria: ", flat(crit))
  res <- list(tag = tag, model = "ordinal", iter = iter, warmup = warmup, do_bridge = TRUE,
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
top_diff <- c("delta", "mu", "tau", "sigma", "mu_new")
top_rating <- c("delta", "beta", "alpha", "tau_alpha", "tau_beta", "sigma", "beta_new", "mean_ai", "mean_human")
top_lkj <- c("delta", "beta", "alpha", "tau_ab", "rho_ab", "sigma", "beta_new")

if (task == "data") {
  # ----- 除外・共有データとの照合・文化的妥当性の読み方の点検・記述統計 -----
  t_task <- Sys.time()
  ex <- h1_read_expert(flags_file)
  res <- list(
    exclusion = h1_exclusion_table(ex$persons),
    unassigned = ex$unassigned,
    n_rows_raw = ex$n_rows_raw,
    cross_check = h1_cross_check(ex$long, ex$persons, rating_long_file),
    cv_check = h1_cv_check(ex$long),
    # 観点ごとの対の数・人数・尺度数(主分析の向き)
    pairs = purrr::imap(aspect_map, function(a, nm) {
      df <- h3_make_diff(h1_prepare(ex$long, a))
      tibble::tibble(aspect = a, n_pairs = nrow(df), n_persons = dplyr::n_distinct(df$subject_id),
                     n_scales = dplyr::n_distinct(df$scale_id), min_pairs_per_scale = min(table(df$scale_id)),
                     max_pairs_per_scale = max(table(df$scale_id)))
    }) %>% dplyr::bind_rows(),
    # 観点ごとの記述統計(尺度×版。文化的妥当性は記録された値のまま)
    describe = purrr::imap(aspect_map, function(a, nm) {
      h3_describe(h1_prepare(ex$long, a)) %>% dplyr::mutate(aspect = a, .before = 1)
    }) %>% dplyr::bind_rows(),
    # 文化的妥当性の2つの扱いの記述(全体の差の平均)
    cv_modes = purrr::map(c("asis", "reverse"), function(m) {
      df <- h3_make_diff(h1_prepare(ex$long, "cultural_validity", cv_mode = m))
      tibble::tibble(cv_mode = m, n_pairs = nrow(df), mean_d = mean(df$d), sd_d = stats::sd(df$d))
    }) %>% dplyr::bind_rows()
  )
  saveRDS(res, file.path(out_dir, "data_checks.rds"))
  readr::write_csv(res$exclusion, file.path(out_dir, "exclusion.csv"))
  readr::write_csv(res$pairs, file.path(out_dir, "pairs.csv"))
  readr::write_csv(res$describe, file.path(out_dir, "describe.csv"))
  readr::write_csv(res$cv_check$pooled, file.path(out_dir, "cv_check_pooled.csv"))
  readr::write_csv(res$cv_check$freq, file.path(out_dir, "cv_check_freq.csv"))
  readr::write_csv(res$cv_modes, file.path(out_dir, "cv_modes.csv"))
  log_line("cross check: ", flat(res$cross_check))
  log_line("done task data in ", round(secs(t_task), 1), " s")
} else if (task != "collect") {
  t_task <- Sys.time()
  aspect_name <- aspect_map[[aspect_short]]
  log_line("start ", tag, " (aspect = ", aspect_name, ", iter = ", iter, ", warmup = ", warmup, ")")
  # ----- データ(文化的妥当性の扱いは課題で切り替える) -----
  cv_mode <- if (grepl("^x_cvrev_", task)) "reverse" else "asis"
  if (cv_mode != "asis" && aspect_short != "cv") stop("x_cvrev は cv だけの課題です")
  ex <- h1_read_expert(flags_file)
  wording_long <- h1_prepare(ex$long, aspect_name, cv_mode = cv_mode)
  diff_df <- h3_make_diff(wording_long)
  log_line("data: rating rows = ", nrow(wording_long), ", diff rows = ", nrow(diff_df),
           ", scales = ", dplyr::n_distinct(wording_long$scale_id),
           ", persons = ", dplyr::n_distinct(wording_long$subject_id), ", cv_mode = ", cv_mode)

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

  if (task %in% c("main_diff", "main_rating", "x_cvrev_diff", "x_cvrev_rating")) {
    # ----- 主分析と文化的妥当性を逆転した版(同じ手順) -----
    model_name <- sub("^(main|x_cvrev)_", "", task)
    pd <- pair_data(model_name)
    run_pair(tag, model_name, pd$h1, pd$h0,
             top_pars = if (model_name == "diff") top_diff else top_rating,
             scale_pars = if (model_name == "diff") "mu_scale" else c("beta_scale", "alpha_scale"),
             keep_scale_draws = grepl("^main", task))
  } else if (grepl("^x_rscale_", task)) {
    # ----- r_scale を変える(H1 だけ推定し，H0 は同じ観点の主分析のものを使う) -----
    parts <- strsplit(sub("^x_rscale_", "", task), "_")[[1]]
    model_name <- parts[1]
    pd <- pair_data(model_name, r_scale = rscale_values[[parts[2]]])
    run_pair(tag, model_name, pd$h1, pd$h0,
             top_pars = if (model_name == "diff") top_diff else top_rating,
             h0_from = paste0(aspect_short, "_main_", model_name))
  } else if (grepl("^x_prior_", task)) {
    # ----- 分散の事前分布の幅を半分・倍(尺度間 SD と残差 SD の両方) -----
    parts <- strsplit(sub("^x_prior_", "", task), "_")[[1]]
    k <- prior_mult[[parts[1]]]
    model_name <- parts[2]
    pd <- pair_data(model_name, s_tau = 0.5 * k, s_sigma = 1 * k)
    run_pair(tag, model_name, pd$h1, pd$h0, top_pars = if (model_name == "diff") top_diff else top_rating)
  } else if (task == "x_lkj") {
    # ----- 評定モデルに α_j と β_j の相関(LKJ η = 1) -----
    pd <- pair_data("rating", lkj_eta = 1)
    run_pair(tag, "rating_lkj", pd$h1, pd$h0, top_pars = top_lkj)
  } else if (task == "x_person") {
    # ----- 差得点モデルに回答者の効果(仮説3で BF が安定しなかったので推定と Savage–Dickey のみ) -----
    pd <- pair_data("diff", use_person = 1)
    run_pair(tag, "diff", pd$h1, pd$h0, top_pars = c(top_diff, "tau_person", "sd_unit"), do_bridge = FALSE)
  } else if (task == "x_order") {
    # ----- 差得点モデルに先に評価した版の項 ＋ 記述統計 -----
    saveRDS(h3_order_describe(diff_df), file.path(out_dir, sprintf("order_describe_%s.rds", aspect_short)))
    pd <- pair_data("diff", use_order = 1)
    run_pair(tag, "diff", pd$h1, pd$h0, top_pars = c(top_diff, "gamma"))
  } else if (task %in% c("x_ordinal_h1", "x_ordinal_h0")) {
    # ----- 累積プロビット(文化的妥当性は記録された値の順序) -----
    run_ordinal_half(paste0(aspect_short, "_x_ordinal"), wording_long, sub("^x_ordinal_", "", task))
  } else if (task == "x_ordinal_merge") {
    run_ordinal_merge(paste0(aspect_short, "_x_ordinal"))
  } else if (task == "scale_bf") {
    # ----- 尺度ごとの参考記録(21本)。r_scale の探索値も同時に出す -----
    rs_all <- c(main = H3_R_SCALE, rscale_values)
    one_mode <- function(wl, mode_label) {
      purrr::imap(rs_all, function(r, nm) {
        dplyr::bind_rows(
          h1_scale_bf(wl, "diff", r_scale = r) %>% dplyr::mutate(model = "diff", .before = 1),
          h1_scale_bf(wl, "rating", r_scale = r) %>% dplyr::mutate(model = "rating", .before = 1)
        ) %>% dplyr::mutate(r_label = nm, r_scale = r, .before = 1)
      }) %>% dplyr::bind_rows() %>% dplyr::mutate(cv_mode = mode_label, .before = 1)
    }
    res <- one_mode(wording_long, "asis")
    # 文化的妥当性は 6 − 値 に逆転した版も主の r だけで出す
    if (aspect_short == "cv") {
      rs_all <- c(main = H3_R_SCALE)
      res <- dplyr::bind_rows(res, one_mode(h1_prepare(ex$long, aspect_name, "reverse"), "reverse"))
    }
    res <- res %>% dplyr::mutate(aspect = aspect_name, .before = 1)
    saveRDS(res, file.path(out_dir, sprintf("scale_bf_%s.rds", aspect_short)))
  }
  log_line("done ", tag, " in ", round(secs(t_task), 1), " s")
}


# ============================================================
# 集計(要約の rds と csv を作る)
# ============================================================
if (task == "collect") {
  files <- list.files(out_dir, pattern = "^pair_.*\\.rds$", full.names = TRUE)
  pairs <- purrr::map(files, readRDS) %>% purrr::set_names(sub("^pair_(.*)\\.rds$", "\\1", basename(files)))
  delta_row <- function(p) {
    out <- p$top %>% dplyr::filter(hyp == "H1", par %in% c("delta[1]", "b_x")) %>%
      dplyr::select(delta_mean = mean, delta_hdi_low = hdi_low, delta_hdi_high = hdi_high, p_rope)
    return(out)
  }
  raw_row <- function(p) {
    out <- p$top %>% dplyr::filter(hyp == "H1", par %in% c("mu", "beta")) %>%
      dplyr::select(raw_mean = mean, raw_hdi_low = hdi_low, raw_hdi_high = hdi_high)
    if (nrow(out) == 0) out <- tibble::tibble(raw_mean = NA_real_, raw_hdi_low = NA_real_, raw_hdi_high = NA_real_)
    return(out)
  }
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
    dplyr::mutate(
      aspect = aspect_map[sub("_.*$", "", tag)],
      bf_mark = bf_mark(bf01),
      hdi_rope = dplyr::case_when(delta_hdi_low > H3_ROPE[1] & delta_hdi_high < H3_ROPE[2] ~ "inside",
                                  delta_hdi_high < H3_ROPE[1] | delta_hdi_low > H3_ROPE[2] ~ "outside",
                                  TRUE ~ "overlap"),
      conv_pass = max_rhat < 1.01 & min_ess_bulk > 1000 & min_ess_tail > 1000 & n_divergent == 0,
      .before = 1
    )
  readr::write_csv(summary_tbl, file.path(out_dir, "summary_bf_delta.csv"))
  top_all <- purrr::imap(pairs, function(p, nm) dplyr::mutate(p$top, tag = nm, .before = 1)) %>% dplyr::bind_rows()
  readr::write_csv(top_all, file.path(out_dir, "top_parameters.csv"))
  is_main <- grepl("^[a-z]+_main_", names(pairs))
  scales_all <- purrr::imap(pairs[is_main], function(p, nm) dplyr::mutate(p$scales, tag = nm, .before = 1)) %>%
    dplyr::bind_rows()
  readr::write_csv(scales_all, file.path(out_dir, "main_scale_posteriors.csv"))
  bridge_all <- purrr::imap(pairs, function(p, nm) dplyr::mutate(p$bridge, tag = nm, .before = 1)) %>% dplyr::bind_rows()
  readr::write_csv(bridge_all, file.path(out_dir, "bridge_reps.csv"))
  conv_all <- purrr::imap(pairs, function(p, nm) dplyr::mutate(p$conv, tag = nm, .before = 1)) %>% dplyr::bind_rows()
  readr::write_csv(conv_all, file.path(out_dir, "convergence.csv"))
  times_all <- purrr::imap(pairs, function(p, nm) dplyr::mutate(p$times, tag = nm, .before = 1)) %>% dplyr::bind_rows()
  readr::write_csv(times_all, file.path(out_dir, "times.csv"))
  # 尺度ごとの参考記録
  sbf_files <- list.files(out_dir, pattern = "^scale_bf_.*\\.rds$", full.names = TRUE)
  sbf <- purrr::map(sbf_files, readRDS) %>% dplyr::bind_rows()
  if (nrow(sbf) > 0) {
    sbf <- sbf %>%
      dplyr::mutate(bf_mark = bf_mark(bf01),
                    hdi_rope = dplyr::case_when(is.na(delta_hdi_low) ~ NA_character_,
                                                delta_hdi_low > H3_ROPE[1] & delta_hdi_high < H3_ROPE[2] ~ "inside",
                                                delta_hdi_high < H3_ROPE[1] | delta_hdi_low > H3_ROPE[2] ~ "outside",
                                                TRUE ~ "overlap"))
    readr::write_csv(sbf, file.path(out_dir, "scale_bf.csv"))
  }
  # 先に評価した版ごとの記述統計
  od_files <- list.files(out_dir, pattern = "^order_describe_.*\\.rds$", full.names = TRUE)
  od <- purrr::map(od_files, function(f) {
    readRDS(f) %>% dplyr::mutate(aspect = sub("^order_describe_(.*)\\.rds$", "\\1", basename(f)), .before = 1)
  }) %>% dplyr::bind_rows()
  if (nrow(od) > 0) readr::write_csv(od, file.path(out_dir, "order_describe.csv"))
  saveRDS(list(summary = summary_tbl, top = top_all, scales = scales_all, bridge = bridge_all,
               conv = conv_all, times = times_all, scale_bf = sbf, order_describe = od,
               scale_draws = purrr::map(pairs[is_main], "scale_draws")),
          file.path(out_dir, "h1rep_main_results.rds"))
}
