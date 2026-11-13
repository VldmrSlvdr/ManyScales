# 仮説2 平行分析の乱数の種による揺れの集計(主分析＝除外基準3なし) ----------------
# 2026-10-07
# 計算: Al-khwa ~/manyscales_h2_wo3/ で run_pa_seed.R <種>(qmd の pa-compute チャンクの
#       set.seed と保存先だけを差し替えて実行。mc.cores = 1・n.iter = 1000・各プロセス1コア)
# 種: 20261003(qmd と同じ。既存の results/h2_parallel.rds)と 20261004〜20261012
# 入力: results/h2_parallel.rds と results/h2_pa_seeds/h2_parallel_seed*.rds
# 出力: results/h2_pa_seeds/h2_pa_seed_stability_*.csv
# 使い方: primary/ を作業ディレクトリにして Rscript R/h2_pa_seed_stability.R

pacman::p_load(tidyverse)

# 設定 ------------------------------------------------------------------------
seed_main <- 20261003L
dir_seeds <- "results/h2_pa_seeds"

# 原論文の因子数(qmd の pa-judge と同じ定義。TIVS = 11 を除く) ------------------
key_h2 <- readr::read_csv("R/h2_item_key_draft.csv", show_col_types = FALSE) |>
  dplyr::filter(!scale_id %in% c(11L))
n_factor_original <- key_h2 |>
  dplyr::summarise(abbr = dplyr::first(abbr), n_factor_original = dplyr::n_distinct(factor), .by = scale_id)

# 10通りの結果を読み込む --------------------------------------------------------
read_seed <- function(path, seed) {
  out <- readRDS(path) |>
    dplyr::mutate(seed = seed)
  return(out)
}
files_other <- list.files(dir_seeds, pattern = "^h2_parallel_seed[0-9]+\\.rds$", full.names = TRUE)
seeds_other <- as.integer(stringr::str_extract(basename(files_other), "[0-9]{8}"))
pa_all <- dplyr::bind_rows(
  read_seed("results/h2_parallel.rds", seed_main),
  purrr::map2(files_other, seeds_other, read_seed)
)
stopifnot(dplyr::n_distinct(pa_all$seed) == 10L)

# 縦長にする: 種 × 尺度 × 相関(ポリコリック/ピアソン) × 方式(因子分析版/主成分版) × 版
pa_long <- pa_all |>
  tidyr::pivot_longer(c(nfact_fa, ncomp_pc), names_to = "method", values_to = "n") |>
  dplyr::mutate(method = dplyr::if_else(method == "nfact_fa", "FA", "PC")) |>
  dplyr::left_join(n_factor_original, by = "scale_id")

# 種ごとの判定(qmd と同じく「両版の因子数が一致」。あわせて原論文の因子数との一致も) ----
pa_judge <- pa_long |>
  dplyr::select(seed, scale_id, abbr, n_factor_original, cor_type, method, version, n) |>
  tidyr::pivot_wider(names_from = version, values_from = n) |>
  dplyr::mutate(
    same_versions = human == ai,
    human_eq_orig = human == n_factor_original,
    ai_eq_orig = ai == n_factor_original
  )

# 因子数の分布を「3×7, 4×3」の形の文字列にする
dist_string <- function(x) {
  tab <- sort(table(x), decreasing = TRUE)
  out <- paste0(names(tab), "×", as.integer(tab), collapse = ", ")
  return(out)
}

# 尺度 × 相関 × 方式ごとの集計(10回) ----------------------------------------------
pa_stability <- pa_judge |>
  dplyr::summarise(
    n_factor_original = dplyr::first(n_factor_original),
    human_main = human[seed == seed_main],
    ai_main = ai[seed == seed_main],
    human_dist = dist_string(human),
    ai_dist = dist_string(ai),
    same_main = same_versions[seed == seed_main],
    same_count = sum(same_versions),
    human_orig_count = sum(human_eq_orig),
    ai_orig_count = sum(ai_eq_orig),
    .by = c(cor_type, method, scale_id, abbr)
  ) |>
  dplyr::mutate(
    # 両版一致の判定が10回とも同じか
    stable = same_count %in% c(0L, 10L),
    # 10回の多数派の判定(5対5は「同数」)
    same_majority = dplyr::case_when(
      same_count > 5L ~ "一致",
      same_count < 5L ~ "不一致",
      TRUE ~ "同数"
    ),
    main_matches_majority = dplyr::if_else(same_main, "一致", "不一致") == same_majority
  ) |>
  dplyr::arrange(cor_type, method, scale_id)

# 種ごとの一致数(21尺度中) ---------------------------------------------------------
pa_count_by_seed <- pa_judge |>
  dplyr::summarise(
    n_same = sum(same_versions),
    n_human_orig = sum(human_eq_orig),
    n_ai_orig = sum(ai_eq_orig),
    .by = c(cor_type, method, seed)
  ) |>
  dplyr::arrange(cor_type, method, seed)

# 一致数の分布の要約(相関 × 方式ごと)
pa_count_summary <- pa_count_by_seed |>
  dplyr::summarise(
    main = n_same[seed == seed_main],
    min = min(n_same),
    max = max(n_same),
    dist = dist_string(n_same),
    # 多数派判定(尺度ごとに10回の多数派をとった場合)の一致数
    .by = c(cor_type, method)
  ) |>
  dplyr::left_join(
    pa_stability |>
      dplyr::summarise(n_majority = sum(same_majority == "一致"), n_unstable = sum(!stable), .by = c(cor_type, method)),
    by = c("cor_type", "method")
  )

# 保存 ------------------------------------------------------------------------
readr::write_csv(pa_stability, file.path(dir_seeds, "h2_pa_seed_stability_by_scale.csv"))
readr::write_csv(pa_count_by_seed, file.path(dir_seeds, "h2_pa_seed_stability_by_seed.csv"))
readr::write_csv(pa_count_summary, file.path(dir_seeds, "h2_pa_seed_stability_summary.csv"))
