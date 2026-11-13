rm(list = ls())
pacman::p_load(tidyverse, readxl)

# 仮説2の準備: 項目と因子の対応表(キー)の作成と点検 ----------------------
# 1. ManyScales対象尺度.xlsx の尺度ごとのシートから，項目・逆転印・因子を抜き出す
# 2. xlsxの訳文と調査票(生データ2行目の設問文)を突き合わせ，項目番号の対応を確かめる
# 3. 逆転印つき項目が，データの段階で既に逆転済みかを項目間相関で確かめる
# 4. 因子割り当てがデータと矛盾しないかを，項目と各因子の合計との相関で確かめる
# 5. 対応表の下書きを R/h2_item_key_draft.csv に書き出す
# 結果は print せず，オブジェクトとして残す(RStudioのEnvironmentで見る)

load("data/4_Rdata/ManyScales_long.Rdata")
path_xlsx <- "../../../translation/materials/ManyScales対象尺度.xlsx"
path_raw <- "data/0_raw_data/2026_ManyScales_一般_2026年7月29日_06.39.tsv"

# 1. xlsxから項目・逆転印・因子を抜き出す ---------------------------------
# 尺度ごとのシート名は「ID 略称」(例: "16 MPS")。0(サンプル)とx付き(不採用)は除く
# 「項目」と書かれた行の次から，英語原文(2列目)が空でない行が項目
# 列: 1=項目番号 2=英語原文 3=人手訳 4=逆転印(r) 5=因子 7=人手訳の調整版 9=AI訳 11=AI訳の調整版
read_scale_sheet <- function(sheet) {
  df <- readxl::read_excel(path_xlsx, sheet = sheet, col_names = FALSE,
                           col_types = "text", .name_repair = "minimal")
  names(df) <- paste0("c", seq_along(df))
  start <- which(df$c1 == "項目")[1]
  items <- df[(start + 1):nrow(df), ] |>
    dplyr::filter(!is.na(c2)) |>
    dplyr::transmute(
      scale_id = as.integer(stringr::str_extract(sheet, "^[0-9]+")),
      abbr = stringr::str_remove(sheet, "^[0-9]+ "),
      item = as.integer(as.numeric(c1)),
      english = stringr::str_squish(c2),
      human_ja = c3, human_recon = c7, ai_ja = c9, ai_recon = c11,
      reverse_marked = as.integer(dplyr::coalesce(c4, "") == "r"),
      factor_xlsx = dplyr::coalesce(c5, "")
    )
  return(items)
}

scale_sheets <- readxl::excel_sheets(path_xlsx) |>
  stringr::str_subset("^[0-9]+ ") |>
  stringr::str_subset("^0 ", negate = TRUE)

key_xlsx <- scale_sheets |>
  purrr::map(read_scale_sheet) |>
  dplyr::bind_rows() |>
  # SCSSの39-41はattention check(データからは除去済み)
  dplyr::filter(!(scale_id == 7 & item > 38))

# 尺度ごとの項目数・因子数・逆転印の数
# n_factor           = 尺度対応表.xlsx の因子数(原論文の因子数)
# n_item_with_factor = xlsxの因子欄に記入がある項目の数(0なら因子欄が空)
# n_factor_xlsx      = xlsxの因子欄に書かれた因子の種類の数
# n_factor と n_factor_xlsx が食い違う尺度は，因子欄の書き漏れか上位因子だけの記入
# 表の下に「注」の行があるので，IDが数字の行だけを使う
n_factor_table <- readxl::read_excel("../../../translation/materials/尺度対応表.xlsx") |>
  dplyr::filter(stringr::str_detect(ID, "^[0-9]+$")) |>
  dplyr::transmute(scale_id = as.integer(ID), n_factor = as.integer(因子数))

key_summary <- key_xlsx |>
  dplyr::group_by(scale_id, abbr) |>
  dplyr::summarise(
    n_item = dplyr::n(),
    n_item_with_factor = sum(factor_xlsx != ""),
    n_factor_xlsx = dplyr::n_distinct(factor_xlsx[factor_xlsx != ""]),
    n_reverse = sum(reverse_marked),
    factors = paste(unique(factor_xlsx), collapse = " / "),
    .groups = "drop"
  ) |>
  dplyr::left_join(n_factor_table, by = "scale_id") |>
  dplyr::relocate(n_factor, .after = n_item)

# 2. xlsxの訳文と調査票の設問文の照合 ------------------------------------
# 生データはQualtricsのUTF-16のTSVで，1行目=列名，2行目=設問文
# 設問文の中に改行があるので，引用符を解釈して読む
raw_head <- utils::read.delim(path_raw, fileEncoding = "UTF-16LE", header = FALSE,
                              nrows = 2, quote = "\"", colClasses = "character")
survey_text <- tibble::tibble(col = unlist(raw_head[1, ]), text = unlist(raw_head[2, ])) |>
  dplyr::filter(stringr::str_detect(col, "^[A-F][0-9]{2}_item[0-9]+_(human|ai)$")) |>
  tidyr::extract(col, c("scale_id", "item", "version"),
                 "^[A-F]([0-9]{2})_item([0-9]+)_(human|ai)$", convert = TRUE) |>
  # MCSはCとDの両方にあるので1つに絞る
  dplyr::distinct(scale_id, item, version, .keep_all = TRUE)

# 空白と句読点を除いてから編集距離で類似度(0-1)を出す
normalize_text <- function(x) {
  return(stringr::str_remove_all(dplyr::coalesce(x, ""), "[\\s　。．.、，,]"))
}
text_similarity <- function(a, b) {
  a <- normalize_text(a)
  b <- normalize_text(b)
  d <- mapply(function(x, y) utils::adist(x, y)[1, 1], a, b)
  return(1 - d / pmax(nchar(a), nchar(b), 1))
}

text_match <- key_xlsx |>
  dplyr::select(scale_id, abbr, item, human_ja, human_recon, ai_ja, ai_recon) |>
  tidyr::pivot_longer(c(human_ja, human_recon, ai_ja, ai_recon),
                      names_to = "source", values_to = "xlsx_text") |>
  dplyr::mutate(version = dplyr::if_else(stringr::str_starts(source, "human"), "human", "ai")) |>
  dplyr::left_join(survey_text, by = c("scale_id", "item", "version")) |>
  dplyr::mutate(sim = text_similarity(text, xlsx_text)) |>
  # 元の訳と調整版のうち，調査票に近いほうを採る
  dplyr::group_by(scale_id, abbr, item, version) |>
  dplyr::summarise(sim = max(sim, na.rm = TRUE), .groups = "drop")

# 類似度が.8未満の項目(0件なら項目番号の対応は確実)
text_mismatch <- text_match |> dplyr::filter(sim < .8)

# 3. 逆転印つき項目は既に逆転済みか ----------------------------------------
# 1尺度・1版の回答を 人×項目 の行列にする
item_matrix <- function(sid, v) {
  m <- citizen_long |>
    dplyr::filter(scale_id == sid, version == v) |>
    dplyr::select(subject_id, item, response) |>
    tidyr::pivot_wider(names_from = item, values_from = response) |>
    dplyr::select(-subject_id) |>
    as.matrix()
  return(m[, order(as.integer(colnames(m)))])
}

# 逆転印つき項目ごとに，同じ因子で逆転印のない項目との平均相関を出す
# 生の回答なら負になるはず。正なら調査システムの段階で逆転済み
# 1因子の尺度は因子欄が空なので，尺度全体を1因子とみなす
# MPSは因子割り当て自体が疑わしい(4.を参照)ので，ここでの判定は参考にとどめる
reverse_check <- key_xlsx |>
  dplyr::filter(reverse_marked == 1) |>
  dplyr::distinct(scale_id) |>
  dplyr::pull(scale_id) |>
  purrr::map(function(sid) {
    k <- key_xlsx |>
      dplyr::filter(scale_id == sid) |>
      dplyr::mutate(fac = dplyr::if_else(factor_xlsx == "", "F1", factor_xlsx))
    res <- purrr::map(c("human", "ai"), function(v) {
      r <- stats::cor(item_matrix(sid, v), use = "pairwise.complete.obs")
      out <- k |>
        dplyr::filter(reverse_marked == 1) |>
        dplyr::mutate(version = v, r_with_forward = purrr::map2_dbl(item, fac, function(i, f) {
          fw <- k$item[k$fac == f & k$reverse_marked == 0]
          # GAAISのNegのように因子の全項目が逆転印つきなら，尺度内の逆転印のない項目と比べる
          if (length(fw) == 0) fw <- k$item[k$reverse_marked == 0]
          return(mean(r[as.character(i), as.character(fw)]))
        }))
      return(out)
    })
    return(dplyr::bind_rows(res))
  }) |>
  dplyr::bind_rows() |>
  dplyr::select(scale_id, abbr, item, fac, version, r_with_forward) |>
  tidyr::pivot_wider(names_from = version, values_from = r_with_forward, names_prefix = "r_") |>
  dplyr::mutate(judge = dplyr::case_when(
    r_human > .10 & r_ai > .10 ~ "逆転済み",
    r_human < -.10 & r_ai < -.10 ~ "未逆転",
    TRUE ~ "判定不能"
  ))

# 個別に目で見た相関行列(本文で例に挙げたもの)
cor_examples <- list(
  SCSS_punishment = stats::cor(item_matrix(7, "human")[, as.character(7:10)], use = "pairwise"),
  GAAIS = stats::cor(item_matrix(22, "human")[, as.character(c(1, 5, 2, 3, 4))], use = "pairwise"),
  rAOTE_human = stats::cor(item_matrix(6, "human"), use = "pairwise"),
  rAOTE_ai = stats::cor(item_matrix(6, "ai"), use = "pairwise"),
  VAST_human = stats::cor(item_matrix(20, "human"), use = "pairwise"),
  MPS_selected = stats::cor(item_matrix(16, "human")[, as.character(c(1, 8, 12, 36, 2, 4, 38, 3, 37))],
                            use = "pairwise")
) |>
  purrr::map(round, 2)

# 4. 因子割り当ての点検 ----------------------------------------------------
# 項目ごとに，自分の因子(自分を除く)の合計との相関と，他の因子の合計との相関を比べる
# データは逆転済みとみなし，ここでは逆転をかけない
factor_check <- key_xlsx |>
  dplyr::filter(factor_xlsx != "") |>
  dplyr::distinct(scale_id) |>
  dplyr::pull(scale_id) |>
  purrr::map(function(sid) {
    k <- key_xlsx |> dplyr::filter(scale_id == sid)
    facs <- unique(k$factor_xlsx)
    res <- purrr::map(c("human", "ai"), function(v) {
      m <- item_matrix(sid, v)[, as.character(k$item)]
      rs <- purrr::map(seq_len(nrow(k)), function(j) {
        r <- purrr::map_dbl(facs, function(f) {
          idx <- setdiff(which(k$factor_xlsx == f), j)
          if (length(idx) == 0) return(NA_real_)
          return(stats::cor(m[, j], rowSums(m[, idx, drop = FALSE]), use = "pairwise"))
        })
        names(r) <- facs
        return(tibble::tibble(
          item = k$item[j], own = k$factor_xlsx[j], r_own = r[[k$factor_xlsx[j]]],
          best = names(r)[which.max(r)], r_best = max(r, na.rm = TRUE)
        ))
      })
      return(dplyr::mutate(dplyr::bind_rows(rs), scale_id = sid, abbr = k$abbr[1], version = v))
    })
    return(dplyr::bind_rows(res))
  }) |>
  dplyr::bind_rows()

# 別の因子の合計のほうが.10以上高い項目
factor_mismatch <- factor_check |>
  dplyr::filter(best != own, r_best - r_own > .10)

# MPS: xlsxの因子(行)と最も相関の高い因子(列)の対応。対角に集まらなければ割り当てが疑わしい
mps_crosstab <- factor_check |>
  dplyr::filter(scale_id == 16, version == "human") |>
  dplyr::count(own, best) |>
  tidyr::pivot_wider(names_from = best, values_from = n, values_fill = 0)

# 5. 対応表の下書きを書き出す ----------------------------------------------
# 1因子の尺度(FOBS, rAOT-E, MESx, FoMO)は F1 とする
# xlsxの因子欄が空(TIVS・MEMI)または誤り(MPS・MOS)の尺度は，原典で確認したキーに差し替える
key_notes <- tibble::tribble(
  ~scale_id, ~item, ~note,
  6L, 4L, "逆転印ありだがデータは未逆転の向き(他項目と負相関)",
  6L, 3L, "他項目とほぼ無相関で向きを判定できない",
  6L, 8L, "他項目とほぼ無相関で向きを判定できない",
  20L, 6L, "他項目とほぼ無相関で向きを判定できない",
  11L, 7L, "他項目とほぼ無相関"
)

# MEMI(ID 18)の項目と因子の対応は，原論文の付録(Hopwood et al., 2021, Appendix)で確認した
# 調査票の項目番号は付録の番号と同じ並び
memi_key <- tibble::tibble(
  scale_id = 18L,
  item = c(1L, 8L, 18L, 19L, 2L, 5L, 9L, 12L, 15L, 3L, 6L, 10L, 13L, 16L, 4L, 7L, 11L, 14L, 17L),
  factor_memi = rep(c("Natural", "Necessary", "Normal", "Nice"), times = c(4, 5, 5, 5))
)

# TIVS(ID 11)の項目と因子の対応は，原論文のTable 1(Gabay et al., 2020)で確認した
# 調査票の項目番号はTable 1の並びと同じ(1-6, 7-12, 13-18, 19-22)
# 原論文の最終モデルは，4因子を方法因子とし1つの一般因子を置く階層モデル
tivs_key <- tibble::tibble(
  scale_id = 11L,
  item = 1:22,
  factor_tivs = rep(c("Need for recognition", "Moral elitism", "Lack of empathy", "Rumination"),
                    times = c(6, 6, 6, 4))
)

# MPS(ID 16)はxlsxの因子欄が原論文と合わない(付録の代表項目15個のうち一致は4個)ので使わない
# 作者公開の採点キー(Hewitt lab「Scoring the Multidimensional Perfectionism Scale」)に差し替える
# 逆転項目18個(2,3,4,8,9,10,12,19,21,24,30,34,36,37,38,43,44,45)はxlsxの逆転印と一致
mps_key <- tibble::tibble(
  scale_id = 16L,
  item = c(1L, 6L, 8L, 12L, 14L, 15L, 17L, 20L, 23L, 28L, 32L, 34L, 36L, 40L, 42L,
           2L, 3L, 4L, 7L, 10L, 16L, 19L, 22L, 24L, 26L, 27L, 29L, 38L, 43L, 45L,
           5L, 9L, 11L, 13L, 18L, 21L, 25L, 30L, 31L, 33L, 35L, 37L, 39L, 41L, 44L),
  factor_mps = rep(c("Self-Oriented", "Other-Oriented", "Socially Prescribed"), each = 15)
)

# MOS(ID 17)はxlsxの因子欄で8・9番が情動反応になっているが，原論文では行動的思考
# 原論文(Giacalone et al., 2025, Study 1 Method)で確認した: 情動反応=MO1・2・3・4・6，行動的思考=MO5・7・8・9
# 調査票の並びは論文の MO1-MO9 と同じ(英文で照合)。データのEFA(2因子)も同じ分かれ方
mos_key <- tibble::tibble(
  scale_id = 17L,
  item = c(1L, 2L, 3L, 4L, 6L, 5L, 7L, 8L, 9L),
  factor_mos = rep(c("moral outrage-emotional reactions", "moral outrage-behavioral thoughts"),
                   times = c(5, 4))
)

# IDAS-II(ID 1)は18下位尺度の採点キー(原著者が配布する IDAS-II Measure の
# 「Composition of the IDAS-II Scales」。佐藤さん経由で入手)で割り当てる
# 調査票の項目番号は配布版の番号と同じ(99項目とも英文で照合)
# xlsxの上位3因子は原論文の下位尺度得点のEFAによるもので項目単位のモデルではないので使わない
# 合成尺度 General Depression(20項目，他の下位尺度と重なる)はここには入れない
# 27・64番は General Depression の中でだけ逆転。Well-Being の中では逆転しない(データも未逆転)
idas_scales <- list(
  "Dysphoria" = c(2, 5, 8, 9, 21, 31, 40, 48, 57, 61),
  "Lassitude" = c(6, 29, 30, 43, 54, 55),
  "Insomnia" = c(4, 11, 17, 25, 36, 51),
  "Suicidality" = c(13, 22, 33, 38, 46, 52),
  "Appetite Loss" = c(1, 26, 60),
  "Appetite Gain" = c(19, 24, 63),
  "Well-Being" = c(3, 10, 23, 27, 50, 53, 59, 64),
  "Ill Temper" = c(12, 35, 37, 44, 62),
  "Mania" = c(67, 71, 77, 83, 87),
  "Euphoria" = c(72, 78, 88, 92, 97),
  "Panic" = c(7, 16, 32, 39, 45, 49, 56, 58),
  "Social Anxiety" = c(15, 18, 20, 41, 47, 99),
  "Claustrophobia" = c(74, 80, 84, 90, 94),
  "Traumatic Intrusions" = c(14, 28, 34, 42),
  "Traumatic Avoidance" = c(73, 79, 89, 93),
  "Checking" = c(68, 75, 81),
  "Ordering" = c(65, 69, 82, 85, 95),
  "Cleaning" = c(66, 70, 76, 86, 91, 96, 98)
)
idas_key <- tibble::tibble(
  scale_id = 1L,
  item = as.integer(unlist(idas_scales)),
  factor_idas = rep(names(idas_scales), lengths(idas_scales))
)
stopifnot(nrow(idas_key) == 99, setequal(idas_key$item, 1:99))

# 階層のある尺度は1次因子(factor)と上位因子(factor_upper)を分けて持つ
# NARQ(ID 3)は原論文のTable 1(Back et al., 2013)で確認した。調査票の並びは論文の番号と同じ
# 原論文のCFAは，6つの1次因子(各3項目)→相関する2つの2次因子
narq_key <- tibble::tibble(
  scale_id = 3L,
  item = c(1L, 2L, 8L, 3L, 5L, 15L, 7L, 16L, 18L, 13L, 14L, 17L, 6L, 9L, 10L, 4L, 11L, 12L),
  facet = rep(c("Grandiosity", "Uniqueness", "Charmingness",
                "Devaluation", "Supremacy", "Aggressiveness"), each = 3),
  upper = rep(c("Admiration", "Rivalry"), each = 9)
)

# BTPS(ID 5)は原論文の補足資料(Smith et al., 2016)の項目一覧・採点方法で確認した
# 調査票の並びは論文の番号と違う: 論文の3番(SOP3)が調査票の29番にあり，論文の4-29番が調査票の3-28番に繰り上がる
# そのため英文の中身で照合した(45項目とも一致)。xlsxの上位因子は位置で振られていて10・28・29番が誤り
btps_key <- tibble::tibble(
  scale_id = 5L,
  item = c(1:4, 29, 5:9, 10:14, 15:19, 20:23, 24:27, 28, 30:33, 34:37, 38:41, 42:45),
  facet = rep(c("SOP", "SWC", "COM", "DAA", "SC", "SPP", "OOP", "HC", "ENT", "GRAN"),
              times = c(5, 5, 5, 5, 4, 4, 5, 4, 4, 4))
) |>
  dplyr::mutate(
    item = as.integer(item),
    upper = dplyr::case_when(
      facet %in% c("SOP", "SWC") ~ "Rigid",
      facet %in% c("COM", "DAA", "SC", "SPP") ~ "Self-critical",
      TRUE ~ "Narcissistic"
    )
  )

hier_key <- dplyr::bind_rows(narq_key, btps_key)

key_draft <- key_xlsx |>
  dplyr::left_join(memi_key, by = c("scale_id", "item")) |>
  dplyr::left_join(tivs_key, by = c("scale_id", "item")) |>
  dplyr::left_join(mps_key, by = c("scale_id", "item")) |>
  dplyr::left_join(mos_key, by = c("scale_id", "item")) |>
  dplyr::left_join(idas_key, by = c("scale_id", "item")) |>
  dplyr::left_join(hier_key, by = c("scale_id", "item")) |>
  dplyr::mutate(
    factor = dplyr::case_when(
      scale_id %in% c(2, 6, 12, 15) ~ "F1",
      scale_id == 18 ~ factor_memi,
      scale_id == 11 ~ factor_tivs,
      scale_id == 16 ~ factor_mps,
      scale_id == 17 ~ factor_mos,
      scale_id == 1 ~ factor_idas,
      scale_id %in% c(3, 5) ~ facet,
      TRUE ~ factor_xlsx
    ),
    factor_upper = dplyr::case_when(
      scale_id %in% c(3, 5) ~ upper,
      # TCI(ID 13)は4因子が2つの区分に分かれ，総合得点は Other-benefiting - Self-defeating
      scale_id == 13 & grepl("Other-benef", factor) ~ "Other-benefiting",
      scale_id == 13 & grepl("Self-defeating", factor) ~ "Self-defeating",
      scale_id == 11 ~ "TIV",
      TRUE ~ NA_character_
    )
  ) |>
  dplyr::left_join(key_notes, by = c("scale_id", "item")) |>
  dplyr::mutate(note = dplyr::case_when(
    scale_id == 16 ~ "因子は作者公開の採点キーによる(xlsxの因子欄は誤り)",
    scale_id == 17 & item %in% c(8L, 9L) ~ "因子は原論文による(xlsxは情動反応で誤り)",
    scale_id == 1 ~ "因子は原著者配布の採点キーによる(xlsxは上位3因子のみ)",
    TRUE ~ dplyr::coalesce(note, "")
  )) |>
  dplyr::select(scale_id, abbr, item, factor, factor_upper, reverse_marked, english, note)

readr::write_csv(key_draft, "R/h2_item_key_draft.csv")
