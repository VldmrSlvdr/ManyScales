# ============================================================
# 仮説3 独立再現の関数
#   データは引数で受け取り，数値(推定値・HDI・BF01・ROPEに入る確率)だけを返す
#   BF > 3 などの閾値による分類と組み合わせ表は qmd 側で作る(判定は入れない)
# ============================================================

# ----- パッケージ -----
pacman::p_load(here, tidyverse, readr, rstan, bridgesampling, BayesFactor,
               bayestestR, logspline, posterior, brms)

# ----- 既定値 -----
# δ の Cauchy 事前分布の尺度(IPA)
H3_R_SCALE <- sqrt(2) / 2
# ROPE(IPA のデザイン分析の SESOI d = 0.2 を δ の上で使う)
H3_ROPE <- c(-0.2, 0.2)
# 乱数の種
H3_SEED <- 20261006


# ============================================================
# データ
# ============================================================

# 処理済みCSVからwording評定を縦長(人×尺度×版)で取り出す
h3_read_wording <- function(data_dir, sets = LETTERS[1:6], exclude_scales = 11) {
  # セットごとに読み込んで縦長にしたものを積み重ねる
  out <- purrr::map(sets, function(s) {
    # セットのCSVを読む(SSOTなので読むだけで書き戻さない)
    d <- readr::read_csv(file.path(data_dir, sprintf("ManyScales_市民_processed_%s.csv", s)),
                         show_col_types = FALSE)
    # 被験者IDを「セット_行番号」で振る(仮説2のlong形式と同じ規則)
    d <- d %>% dplyr::mutate(subject_id = sprintf("%s_%03d", s, dplyr::row_number()))
    # wordingの評定列だけを選ぶ(タイマー列は正規表現の末尾で除く)
    d %>%
      dplyr::select(subject_id, trans_cond,
                    dplyr::matches(sprintf("^%s[0-9]{2}_wording_(human|ai)$", s))) %>%
      # 列名から尺度IDと版を取り出して縦長にする
      tidyr::pivot_longer(-c(subject_id, trans_cond),
                          names_to = c("scale_id", "version"),
                          names_pattern = "^[A-F]([0-9]{2})_wording_(human|ai)$",
                          values_to = "rating") %>%
      # セット名を列に残す
      dplyr::mutate(set = s)
  }) %>%
    dplyr::bind_rows() %>%
    # 尺度IDを整数にする(MCSはCとDの両方から10として入り，1つの尺度にまとまる)
    dplyr::mutate(scale_id = as.integer(scale_id)) %>%
    # 除外する尺度(既定はTIVS = 11)を落とす
    dplyr::filter(!scale_id %in% exclude_scales) %>%
    # MCSに回答していない人(男性以外)の欠測を落とす。値の向きは変えない(5 = 理解しやすい)
    dplyr::filter(!is.na(rating))
  # 縦長のデータを返す
  return(out)
}

# 人×尺度ごとに差得点(AI版 − 人手版)を作る
h3_make_diff <- function(wording_long) {
  out <- wording_long %>%
    # 版を横に並べる
    tidyr::pivot_wider(id_cols = c(set, subject_id, trans_cond, scale_id),
                       names_from = version, values_from = rating) %>%
    # 差得点を作る(正ならAI版の方が理解しやすいと評定された)
    dplyr::mutate(d = ai - human,
                  # 先に答えた版を ±1/2 で表す(探索パートの順序の項で使う)
                  ai_first = dplyr::if_else(trans_cond == "ai", 0.5, -0.5))
  # 差得点のデータを返す
  return(out)
}

# 尺度×版の記述統計(表1)
h3_describe <- function(wording_long) {
  out <- wording_long %>%
    dplyr::group_by(scale_id, version) %>%
    # 人数・平均・SD・中央値と1〜5の度数を出す
    dplyr::summarise(n = dplyr::n(), mean = mean(rating), sd = stats::sd(rating),
                     median = stats::median(rating),
                     n1 = sum(rating == 1), n2 = sum(rating == 2), n3 = sum(rating == 3),
                     n4 = sum(rating == 4), n5 = sum(rating == 5), .groups = "drop")
  # 記述統計の表を返す
  return(out)
}


# ============================================================
# Stan に渡すデータ
# ============================================================

# 差得点モデルに渡すリストを作る
h3_stan_data_diff <- function(diff_df, r_scale = H3_R_SCALE, s_tau = 0.5, s_sigma = 1,
                              fix_zero = 0, use_person = 0, use_order = 0) {
  # 尺度IDを 1..J に振り直す(対応は scale_ids に残す)
  scale_ids <- sort(unique(diff_df$scale_id))
  # 回答者IDを 1..I に振り直す
  person_ids <- sort(unique(diff_df$subject_id))
  out <- list(
    N = nrow(diff_df),                                     # 観測の数
    J = length(scale_ids),                                 # 尺度の数
    scale = match(diff_df$scale_id, scale_ids),            # 尺度番号
    d = diff_df$d,                                         # 差得点
    r_scale = r_scale,                                     # Cauchy の尺度
    s_tau = s_tau,                                         # 尺度間 SD の事前分布の尺度
    s_sigma = s_sigma,                                     # 残差 SD の事前分布の尺度
    fix_zero = fix_zero,                                   # H0 か H1 か
    use_person = use_person,                               # 回答者の効果の有無
    I = length(person_ids),                                # 回答者の数
    person = match(diff_df$subject_id, person_ids),        # 回答者番号
    use_order = use_order,                                 # 順序の項の有無
    ai_first = diff_df$ai_first,                           # 先に答えた版
    scale_ids = scale_ids                                  # 元の尺度ID(Stan は使わない)
  )
  # リストを返す
  return(out)
}

# 評定モデル(相関なし・LKJ版の両方)に渡すリストを作る
h3_stan_data_rating <- function(wording_long, r_scale = H3_R_SCALE, s_tau = 0.5, s_sigma = 1,
                                fix_zero = 0, lkj_eta = 1) {
  # 尺度IDを 1..J に振り直す
  scale_ids <- sort(unique(wording_long$scale_id))
  out <- list(
    N = nrow(wording_long),                                        # 観測の数
    J = length(scale_ids),                                         # 尺度の数
    scale = match(wording_long$scale_id, scale_ids),               # 尺度番号
    x = dplyr::if_else(wording_long$version == "ai", 0.5, -0.5),   # 版のコード
    y = wording_long$rating,                                       # 評定
    r_scale = r_scale,                                             # Cauchy の尺度
    s_tau = s_tau,                                                 # 尺度間 SD の事前分布の尺度
    s_sigma = s_sigma,                                             # 残差 SD の事前分布の尺度
    lkj_eta = lkj_eta,                                             # LKJ の形(LKJ版だけが使う)
    fix_zero = fix_zero,                                           # H0 か H1 か
    scale_ids = scale_ids                                          # 元の尺度ID(Stan は使わない)
  )
  # リストを返す
  return(out)
}


# ============================================================
# Stan の実行(Al-khwa で行う)
# ============================================================

# 3つのモデルをコンパイルする(Al-khwa でその場でコンパイルする)
h3_compile_models <- function(stan_dir) {
  out <- list(
    diff = rstan::stan_model(file.path(stan_dir, "h3rep_diff.stan")),              # 差得点モデル
    rating = rstan::stan_model(file.path(stan_dir, "h3rep_rating.stan")),          # 評定モデル
    rating_lkj = rstan::stan_model(file.path(stan_dir, "h3rep_rating_lkj.stan"))   # 評定モデル(相関あり・探索)
  )
  # コンパイル済みのモデルを返す
  return(out)
}

# MCMC を回す(bridgesampling に使うので事後の抽出は多めにする)
h3_fit <- function(model, stan_data, chains = 4, iter = 12000, warmup = 2000,
                   seed = H3_SEED, adapt_delta = 0.95) {
  # Stan が使わない要素(scale_ids)を外して渡す
  sd_in <- stan_data[setdiff(names(stan_data), "scale_ids")]
  # サンプリングする(事後の抽出は 4 × 10,000)
  fit <- rstan::sampling(model, data = sd_in, chains = chains, iter = iter,
                         warmup = warmup, seed = seed, cores = chains,
                         control = list(adapt_delta = adapt_delta))
  # stanfit を返す
  return(fit)
}

# 収束の診断(R̂・バルク/テイル ESS・発散の数)
h3_check_convergence <- function(fit, pars) {
  # 指定したパラメータの抽出を posterior の形にする
  draws <- posterior::as_draws_df(as.array(fit, pars = pars))
  # R̂ と ESS をまとめ，発散の数を全行に付ける
  out <- posterior::summarise_draws(draws, "rhat", "ess_bulk", "ess_tail") %>%
    dplyr::mutate(n_divergent = rstan::get_num_divergent(fit))
  # 診断の表を返す(合否は qmd 側)
  return(out)
}


# ============================================================
# 要約(事後平均値・95% HDI・ROPE に入る確率)
# ============================================================

# 抽出のベクトル1本を要約する(内部用)
h3_summarise_draws <- function(x, ci = 0.95, rope = NULL) {
  # HDI を求める
  h <- bayestestR::hdi(x, ci = ci)
  # 事後平均値と HDI を1行にする
  out <- tibble::tibble(mean = mean(x), hdi_low = h$CI_low, hdi_high = h$CI_high)
  # ROPE が渡されたら，事後の抽出が ROPE に入る割合を足す(δ に対してだけ使う)
  if (!is.null(rope)) out$p_rope <- mean(x > rope[1] & x < rope[2])
  # 要約を返す
  return(out)
}

# 最上位のパラメータ(delta・mu・mu_new など)を要約する
h3_summarise_top <- function(fit, pars, ci = 0.95, rope = H3_ROPE) {
  out <- purrr::map(pars, function(p) {
    # パラメータの事後の抽出を取り出す
    x <- as.vector(rstan::extract(fit, pars = p)[[1]])
    # δ にだけ ROPE に入る確率を付ける
    h3_summarise_draws(x, ci = ci, rope = if (p == "delta") rope else NULL) %>%
      dplyr::mutate(par = p, .before = 1)
  }) %>% dplyr::bind_rows()
  # 要約を返す
  return(out)
}

# 尺度ごとのパラメータ(mu_scale・beta_scale)を要約する(図用)
h3_summarise_scales <- function(fit, par, scale_ids, ci = 0.95) {
  # 抽出は 抽出数 × 尺度数 の行列
  m <- rstan::extract(fit, pars = par)[[1]]
  out <- purrr::map(seq_along(scale_ids), function(j) {
    # 尺度 j を要約し，元の尺度IDを付ける
    h3_summarise_draws(m[, j], ci = ci) %>% dplyr::mutate(scale_id = scale_ids[j], .before = 1)
  }) %>% dplyr::bind_rows()
  # 尺度ごとの要約を返す
  return(out)
}


# ============================================================
# ベイズファクター
# ============================================================

# bridgesampling で H1 と H0 の周辺尤度を求めて BF01 を出す
h3_bf_bridge <- function(fit_h1, fit_h0, repetitions = 10, seed = H3_SEED) {
  # 乱数を固定する(bridgesampling の推定は乱数に依存する)
  set.seed(seed)
  # H1 の対数周辺尤度(warp3 法・繰り返して推定のばらつきを見る)
  ml1 <- bridgesampling::bridge_sampler(fit_h1, method = "warp3",
                                        repetitions = repetitions, silent = TRUE)
  # H0 の対数周辺尤度
  ml0 <- bridgesampling::bridge_sampler(fit_h0, method = "warp3",
                                        repetitions = repetitions, silent = TRUE)
  # 繰り返しごとの log BF01 = logml(H0) − logml(H1)
  log_bf01 <- ml0$logml - ml1$logml
  out <- tibble::tibble(
    logml_h1 = stats::median(ml1$logml),           # H1 の対数周辺尤度(中央値)
    logml_h0 = stats::median(ml0$logml),           # H0 の対数周辺尤度(中央値)
    bf01 = exp(stats::median(log_bf01)),           # BF01(中央値)
    bf01_min = exp(min(log_bf01)),                 # 繰り返しの最小
    bf01_max = exp(max(log_bf01))                  # 繰り返しの最大
  )
  # BF の表を返す(分類は qmd 側)
  return(out)
}

# Savage–Dickey 比による検算(δ = 0 での事後密度 / 事前密度)
h3_bf_savage_dickey <- function(fit_h1, r_scale = H3_R_SCALE) {
  # δ の事後の抽出を取り出す
  delta <- as.vector(rstan::extract(fit_h1, pars = "delta")[[1]])
  # 事後密度を logspline で滑らかにする
  ls <- logspline::logspline(delta)
  # 0 での事後密度
  post0 <- logspline::dlogspline(0, ls)
  # 0 での事前密度(Cauchy)
  prior0 <- stats::dcauchy(0, location = 0, scale = r_scale)
  # BF01 = 事後密度 / 事前密度
  out <- tibble::tibble(bf01_sd = post0 / prior0)
  # 検算の値を返す
  return(out)
}

# 尺度ごとの参考記録(21本): BayesFactor で BF01 と δ の 95% HDI・ROPE に入る確率を出す
h3_scale_bf <- function(wording_long, model_type = c("diff", "rating"),
                        r_scale = H3_R_SCALE, ci = 0.95, rope = H3_ROPE,
                        iterations = 10000, seed = H3_SEED) {
  # 差得点モデルに対応させるか評定モデルに対応させるか
  model_type <- match.arg(model_type)
  # 乱数を固定する(事後の抽出のため)
  set.seed(seed)
  out <- wording_long %>%
    # 尺度ごとに分ける
    dplyr::group_split(scale_id) %>%
    purrr::map(function(g) {
      # 人ごとに版を横に並べる(対応を保つ)
      w <- tidyr::pivot_wider(g, id_cols = subject_id, names_from = version, values_from = rating)
      # 差得点モデルは対応のある t，評定モデルは2群の t(どちらも回答者の効果なしの構造に合わせる)
      bf <- BayesFactor::ttestBF(x = w$ai, y = w$human,
                                 paired = (model_type == "diff"), rscale = r_scale)
      # 事後の抽出を取る
      post <- BayesFactor::posterior(bf, iterations = iterations, progress = FALSE)
      # 評定の単位の差の列は paired なら "mu"，2群なら "beta (x - y)"
      col <- if (model_type == "diff") "mu" else "beta (x - y)"
      # 標準化した差 δ を要約する(HDI と ROPE はこちらで見る)
      s <- h3_summarise_draws(as.vector(post[, "delta"]), ci = ci, rope = rope)
      # BF01 は BF10 の逆数。評定の単位の差の事後平均も並べる
      tibble::tibble(scale_id = g$scale_id[1], n = nrow(w),
                     bf01 = 1 / BayesFactor::extractBF(bf)$bf,
                     diff_mean = mean(post[, col]),
                     delta_mean = s$mean, delta_hdi_low = s$hdi_low,
                     delta_hdi_high = s$hdi_high, p_rope = s$p_rope)
    }) %>% dplyr::bind_rows()
  # 尺度ごとの参考記録を返す
  return(out)
}


# ============================================================
# 探索パート
# ============================================================

# 累積プロビットモデル(版の効果は潜在尺度の標準化差なので Cauchy をそのまま置ける)
h3_fit_ordinal <- function(wording_long, r_scale = H3_R_SCALE, fix_zero = 0,
                           seed = H3_SEED, chains = 4, iter = 12000, warmup = 2000) {
  # 版のコードと尺度の因子を作る
  dat <- wording_long %>%
    dplyr::mutate(x = dplyr::if_else(version == "ai", 0.5, -0.5),
                  scale_f = factor(scale_id),
                  rating = ordered(rating))
  # H1 は版の固定効果あり，H0 は固定効果なし(尺度ごとの版の差は両方に入れる・相関なし)
  fml <- if (fix_zero == 0) {
    brms::bf(rating ~ 1 + x + (1 + x || scale_f))
  } else {
    brms::bf(rating ~ 1 + (1 + x || scale_f))
  }
  # 事前分布(尺度間 SD に半t，H1 では版の効果に Cauchy)
  pri <- brms::set_prior("student_t(3, 0, 0.5)", class = "sd")
  if (fix_zero == 0) {
    pri <- c(pri, brms::set_prior(sprintf("cauchy(0, %f)", r_scale), class = "b", coef = "x"))
  }
  # 推定する(bridgesampling のため全パラメータを保存する。rstan をバックエンドにする)
  fit <- brms::brm(fml, data = dat, family = brms::cumulative("probit"), prior = pri,
                   chains = chains, iter = iter, warmup = warmup, cores = chains,
                   seed = seed, save_pars = brms::save_pars(all = TRUE), backend = "rstan")
  # brmsfit を返す
  return(fit)
}

# 累積プロビットの BF01(bridgesampling)
h3_bf_ordinal <- function(fit_h1, fit_h0, repetitions = 10, seed = H3_SEED) {
  # 乱数を固定する
  set.seed(seed)
  # 両モデルの対数周辺尤度
  ml1 <- brms::bridge_sampler(fit_h1, method = "warp3", repetitions = repetitions, silent = TRUE)
  ml0 <- brms::bridge_sampler(fit_h0, method = "warp3", repetitions = repetitions, silent = TRUE)
  # 繰り返しごとの log BF01
  log_bf01 <- ml0$logml - ml1$logml
  out <- tibble::tibble(bf01 = exp(stats::median(log_bf01)),
                        bf01_min = exp(min(log_bf01)), bf01_max = exp(max(log_bf01)))
  # BF の表を返す
  return(out)
}

# 累積プロビットの版の効果(δ)を要約する
h3_summarise_ordinal <- function(fit, ci = 0.95, rope = H3_ROPE) {
  # 版の固定効果の事後の抽出を取り出す
  x <- as.vector(posterior::as_draws_df(fit, variable = "b_x")$b_x)
  # 事後平均値・HDI・ROPE に入る確率
  out <- h3_summarise_draws(x, ci = ci, rope = rope) %>% dplyr::mutate(par = "b_x", .before = 1)
  # 要約を返す
  return(out)
}

# 先に答えた版ごとの差得点の記述統計(尺度ごとと全体)
h3_order_describe <- function(diff_df) {
  out <- diff_df %>%
    # 全体の行を作るために尺度IDを NA(= 全体)にした複製を足す
    dplyr::bind_rows(dplyr::mutate(diff_df, scale_id = NA_integer_)) %>%
    dplyr::group_by(scale_id, trans_cond) %>%
    # 人数・差の平均・SD
    dplyr::summarise(n = dplyr::n(), mean_d = mean(d), sd_d = stats::sd(d), .groups = "drop")
  # 記述統計を返す
  return(out)
}
