# 仮説1・3 独立再現と国里版(origin/main 921f801)の突き合わせ
# - こちらの結果ファイルは読むだけで書き換えない
# - 国里版の数値は origin/main の rds/csv から取り出した値(scale_points.csv は複写して results/compare_kunisato/ に置いた)
# - BF の違いの切り分けは，報告された事後平均・95%区間から尤度を正規近似して事前分布だけを入れ替える(Stan は回さない)

# ============================================================
# パッケージ
# ============================================================
pacman::p_load(here, dplyr, readr, tibble, tidyr)

# ============================================================
# 関数
# ============================================================

# 尤度 N(m, s) と事前密度 prior に対する BF01(数値積分)
bf01_normal_lik <- function(m, s, prior) {
  # H0 での尤度(δ = 0)
  num <- stats::dnorm(0, m, s)
  # H1 での周辺尤度
  den <- stats::integrate(function(t) stats::dnorm(t, m, s) * prior(t), -Inf, Inf)$value
  return(num / den)
}

# 事後(事前 N(0, p) のもと) の平均・SD から正規の尤度の平均・SD を戻す
lik_from_post <- function(pm, ps, p) {
  # 精度の差が尤度の精度
  s2 <- 1 / (1 / ps^2 - 1 / p^2)
  return(c(m = pm * s2 / ps^2, s = sqrt(s2)))
}

# 国里版の値に事前分布を入れ替えた BF01 を付ける
kun_prior_swap <- function(tab) {
  # ロジスティック分布の SD(潜在尺度の残差の単位)
  k <- pi / sqrt(3)
  out <- tab %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      ps = (hi - lo) / (2 * stats::qnorm(0.975)),
      m = lik_from_post(pm, ps, 1)[["m"]],
      s = lik_from_post(pm, ps, 1)[["s"]],
      # 国里版の事前分布のまま(報告値の再現の確認)
      bf01_N01 = bf01_normal_lik(m, s, function(t) stats::dnorm(t, 0, 1)),
      # IPA の Cauchy(0, √2/2) を潜在尺度の残差 SD 単位に置いた場合
      bf01_cauchy_resid = bf01_normal_lik(m, s, function(t) stats::dcauchy(t, 0, sqrt(2) / 2 * k)),
      # 同じく，2版の差の SD(残差 SD × √2) 単位に置いた場合(こちらの d_z に近い単位)
      bf01_cauchy_diff = bf01_normal_lik(m, s, function(t) stats::dcauchy(t, 0, sqrt(2) / 2 * k * sqrt(2)))
    ) %>%
    dplyr::ungroup()
  return(out)
}

# こちらの値に国里版に相当する正規事前分布を置いた BF01 を付ける
our_prior_swap <- function(tab) {
  # ロジスティック分布の SD
  k <- pi / sqrt(3)
  out <- tab %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      s = (hi - lo) / (2 * stats::qnorm(0.975)),
      m = pm,
      # こちらの事前分布のまま(報告値との近さの確認)
      bf01_cauchy = bf01_normal_lik(m, s, function(t) stats::dcauchy(t, 0, sqrt(2) / 2)),
      # 国里版の N(0, 1)(対数オッズ)を残差 SD 単位に直したもの
      bf01_N_resid = bf01_normal_lik(m, s, function(t) stats::dnorm(t, 0, 1 / k)),
      # 同じく，差の SD 単位に直したもの
      bf01_N_diff = bf01_normal_lik(m, s, function(t) stats::dnorm(t, 0, 1 / (k * sqrt(2))))
    ) %>%
    dplyr::ungroup()
  return(out)
}

# ============================================================
# 入力(両者の報告値)
# ============================================================

# 国里版: β(人手 − AI，対数オッズ)の事後平均と 95% 区間，S–D の BF01
kun <- tibble::tribble(
  ~id,          ~pm,   ~lo,   ~hi,  ~bf01_reported,
  "H3_除外3なし", 0.33,  0.04,  0.62, 0.636,
  "H3_除外3あり", 0.38,  0.07,  0.68, 0.328,
  "H1_意味",     -0.39, -0.93,  0.13, 1.250,
  "H1_自然",      0.38, -0.14,  0.89, 1.335,
  "H1_文化",     -0.25, -0.84,  0.32, 2.321
)

# こちら: 差得点モデルの δ(AI − 人手)の事後平均と 95% HDI，bridgesampling の BF01
our <- tibble::tribble(
  ~id,          ~pm,    ~lo,    ~hi,   ~bf01_reported,
  "H3_除外3なし", -0.102, -0.201, -0.001, 2.20,  # 2026-10-30 OCQ の除外後の再計算の値(除外前は -0.101, -0.199, 0.000, 2.21)
  "H3_除外3あり", -0.120, -0.224, -0.014, 1.37,  # 同上(除外前は -0.119, -0.224, -0.014, 1.37)
  "H1_意味",      0.135, -0.010,  0.286, 2.12,
  "H1_自然",     -0.118, -0.287,  0.052, 3.67,
  "H1_文化",      0.081, -0.094,  0.260, 6.57
)

# ============================================================
# 事前分布の入れ替え
# ============================================================
kun_swap <- kun_prior_swap(kun)
our_swap <- our_prior_swap(our)

# ============================================================
# 仮説3の尺度ごとの比較(国里版の予測評定差とこちらの階層モデルの μ_j)
# ============================================================
kun_scale <- readr::read_csv(here::here("supplementary", "results", "compare_kunisato", "kunisato_h3_scale_points_without_excl3.csv"),
                             show_col_types = FALSE) %>%
  dplyr::transmute(scale_id, kun_mean = mean, kun_lo = lower95, kun_hi = upper95)
our_scale <- readr::read_csv(here::here("supplementary", "results", "h3rep_main", "main_scale_posteriors.csv"), show_col_types = FALSE) %>%
  dplyr::filter(tag == "main_diff", par == "mu_scale") %>%
  # 向きを国里版(人手 − AI)にそろえる
  dplyr::transmute(scale_id, our_mean = -mean, our_lo = -hdi_high, our_hi = -hdi_low)
scale_cmp <- dplyr::inner_join(kun_scale, our_scale, by = "scale_id") %>%
  dplyr::mutate(
    kun_side = dplyr::case_when(kun_lo > 0 ~ "人手", kun_hi < 0 ~ "AI", TRUE ~ "0を含む"),
    our_side = dplyr::case_when(our_lo > 0 ~ "人手", our_hi < 0 ~ "AI", TRUE ~ "0を含む"),
    same_side = kun_side == our_side
  )
scale_summary <- tibble::tibble(
  n_scales = nrow(scale_cmp),
  r_mean = stats::cor(scale_cmp$kun_mean, scale_cmp$our_mean),
  n_same_side = sum(scale_cmp$same_side),
  differ_ids = paste(scale_cmp$scale_id[!scale_cmp$same_side], collapse = ",")
)

# ============================================================
# 保存
# ============================================================
readr::write_csv(kun_swap, here::here("supplementary", "results", "compare_kunisato", "kunisato_prior_swap.csv"))
readr::write_csv(our_swap, here::here("supplementary", "results", "compare_kunisato", "ours_prior_swap.csv"))
readr::write_csv(scale_cmp, here::here("supplementary", "results", "compare_kunisato", "h3_scale_compare.csv"))
readr::write_csv(scale_summary, here::here("supplementary", "results", "compare_kunisato", "h3_scale_compare_summary.csv"))
