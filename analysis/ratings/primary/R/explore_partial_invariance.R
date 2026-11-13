# 探索的分析: 訳に苦労した項目の部分不変性(作業の記録) ----------------------
# primary/ をカレントにして Rscript R/explore_partial_invariance.R で実行する
# 結果は results/explore_partial_invariance.rds に保存する
pacman::p_load(tidyverse, lavaan, semTools)

# データ・項目と因子の対応表・仮説2の関数を読み込む
load("data/4_Rdata/ManyScales_long.Rdata")
item_key <- readr::read_csv("R/h2_item_key_draft.csv", show_col_types = FALSE)
source("R/hypothesis_2_functions.R")

# 対象項目の一覧 -------------------------------------------------------------
# 翻訳チームが訳に苦労したと挙げた項目(三浦先生のスプレッドシート 10/22 版・6尺度18項目)
# type はコメントの種類(計算の前に決めたもの): A=置き換え，B=通じるかの懸念，C=補足・意訳，D=訳語の選択
# ai_same は AI版が人手版と同じ選択をしたか(調査票で実施した文面を照合)
targets <- tibble::tribble(
  ~scale_id, ~item, ~type, ~ai_same,
  1L, 18L, "D", FALSE,
  1L, 31L, "C", FALSE,
  7L, 19L, "C", FALSE,
  7L, 22L, "C", FALSE,
  7L, 29L, "C", FALSE,
  7L, 31L, "C", FALSE,
  13L, 1L, "B", TRUE,
  13L, 5L, "A", FALSE,
  13L, 8L, "B", TRUE,
  15L, 1L, "D", FALSE,
  15L, 5L, "B", FALSE,
  15L, 7L, "D", FALSE,
  15L, 9L, "D", FALSE,
  19L, 1L, "D", FALSE,
  19L, 6L, "D", FALSE,
  19L, 11L, "C", FALSE,
  19L, 21L, "A", TRUE,
  21L, 29L, "D", FALSE
) |>
  # 対象項目が属する下位因子を対応表から付ける
  dplyr::left_join(dplyr::select(item_key, scale_id, item, abbr, factor), by = c("scale_id", "item"))

# 回答順の表 ------------------------------------------------------------------
# 先に答えた版(trans_cond)は long データに無いので，processed の CSV から取る
# long の人のIDは「セット_行番号」(kosugi0925.R)なので，同じ規則で振って対応させる
order_tbl <- purrr::map_dfr(LETTERS[1:6], \(s) {
  d <- readr::read_csv(sprintf("data/3_data_processed/ManyScales_市民_processed_%s.csv", s), show_col_types = FALSE)
  return(tibble::tibble(subject_id = sprintf("%s_%03d", s, seq_len(nrow(d))), first = d$trans_cond))
})

# 計算 -----------------------------------------------------------------------
# 下位因子ごとに，その下位因子の項目だけに絞った key を渡す(1下位因子 × 2推定法 × 回答順の有無)
units <- targets |>
  dplyr::summarise(items = list(item), .by = c(scale_id, factor))
partial_raw <- tidyr::expand_grid(units, estimator = c("WLSMV", "MLR"), adjust = c(FALSE, TRUE)) |>
  purrr::pmap(\(scale_id, factor, items, estimator, adjust) {
    key_f <- item_key |>
      dplyr::filter(scale_id == !!scale_id, factor == !!factor)
    # adjust = TRUE のときだけ回答順を共変量に入れる(MIMIC)
    ord <- if (adjust) order_tbl else NULL
    return(fit_partial_invariance(citizen_long, key_f, scale_id, items, estimator, order = ord))
  }) |>
  dplyr::bind_rows()

# 対象項目の情報(尺度名・コメントの種類・AI版の選択)を付けて保存する
partial_res <- partial_raw |>
  dplyr::left_join(dplyr::select(targets, scale_id, item, abbr, type, ai_same), by = c("scale_id", "item")) |>
  dplyr::relocate(abbr, .after = scale_id) |>
  dplyr::relocate(type, ai_same, .after = item)
saveRDS(partial_res, "results/explore_partial_invariance.rds")
