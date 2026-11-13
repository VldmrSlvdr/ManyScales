# ============================================================
# 仮説1 独立再現の関数(専門家データ)
#   枠組み(Stan のモデル・要約・BF)は仮説3の独立再現の関数をそのまま使い，
#   ここには専門家データの読み込み・除外・観点ごとのデータの整形と，
#   文化的妥当性の読み方の点検だけを足す
# ============================================================

# ----- パッケージと仮説3の関数 -----
pacman::p_load(here, tidyverse, readr, BayesFactor, bayestestR)
source(here::here("supplementary", "R", "hypothesis_3_replication_functions.R"))

# ----- 既定値 -----
# 観点(choice_fit は使わない)
H1_ASPECTS <- c("semantic", "naturalness", "cultural_validity")
# 注意チェックの正答(data_manual: S_dqs01〜03 は順に 5，2，3，S_ai-trap は 1)
H1_CHECK_KEY <- c(dqs01 = "5", dqs02 = "2", dqs03 = "3", `ai-trap` = "1")
# 適格性の4項目(1 = はい)
H1_ELIGIBILITY <- c("eligibility_native_japanese", "eligibility_scale_paper",
                    "eligibility_english_citation", "eligibility_masters_or_higher")


# ============================================================
# データの読み込みと除外
# ============================================================

# フラグ付きの TSV(Qualtrics の全回答)から，除外を自分で当てて縦長(人×尺度×観点×版)にする
#   除外の基準(IPA と先生の確認5):
#     - 同意(consent = 1)があり，セットが割り当てられている
#     - 適格性の4項目がすべて 1(はい)。1つでも「いいえ」(2)や未回答なら除外
#     - 注意チェック(S_dqs01〜03)と AI トラップ(S_ai-trap)のうち，回答したものに1問でも誤答があれば除外
#       正答は市民データの扱い(data_manual)と同じ 5・2・3・1。
#       空欄は誤答としない(セット A は dqs を条件ごとに1問ずつしか表示しないため。2_分析フラグ付与_専門家.qmd と同じ扱い)
#   評定の欠測はここでは落とさず，観点ごとの整形(h1_prepare)で両版がそろう対だけを残す
h1_read_expert <- function(flags_file, exclude_scales = 11) {
  # すべて文字列で読む(コードの比較を文字列で行うため)
  raw <- readr::read_tsv(flags_file, col_types = readr::cols(.default = readr::col_character()),
                         na = c("", "NA"), progress = FALSE)
  # 元の行番号を残す(被験者IDに使う)
  raw <- raw %>% dplyr::mutate(flags_row = dplyr::row_number())
  # 人ごとの除外の判定を作る
  persons <- raw %>%
    dplyr::filter(!is.na(set_id)) %>%
    dplyr::select(flags_row, set_id, consent, trans_cond, analysis_target, excl1,
                  dplyr::all_of(H1_ELIGIBILITY), dplyr::matches("^[A-F]_(dqs0[1-3]|ai-trap)$"))
  # 自分のセットの注意チェックの回答を縦長にして正誤を判定する
  checks <- persons %>%
    dplyr::select(flags_row, set_id, dplyr::matches("^[A-F]_(dqs0[1-3]|ai-trap)$")) %>%
    tidyr::pivot_longer(-c(flags_row, set_id), names_to = c("check_set", "check"),
                        names_pattern = "^([A-F])_(.*)$", values_to = "answer") %>%
    dplyr::filter(check_set == set_id) %>%
    dplyr::mutate(correct = H1_CHECK_KEY[check],
                  wrong = !is.na(answer) & answer != correct) %>%
    dplyr::group_by(flags_row) %>%
    dplyr::summarise(n_check_answered = sum(!is.na(answer)), n_check_wrong = sum(wrong), .groups = "drop")
  # 適格性と注意チェックをまとめて除外の理由を付ける
  persons <- persons %>%
    dplyr::left_join(checks, by = "flags_row") %>%
    dplyr::mutate(
      consent_ok = !is.na(consent) & consent == "1",
      eligible = dplyr::if_all(dplyr::all_of(H1_ELIGIBILITY), ~ !is.na(.x) & .x == "1"),
      check_ok = n_check_wrong == 0,
      included = consent_ok & eligible & check_ok,
      # 被験者ID(セットとフラグ付き TSV の行番号)
      subject_id = sprintf("pro_%s_r%03d", set_id, flags_row)
    )
  # 評定の列(3観点 × 2版。choice_fit は使わない)を縦長にする
  rating_cols <- sprintf("^[A-F][0-9]{2}_(%s)_(human|ai)$", paste(H1_ASPECTS, collapse = "|"))
  long <- raw %>%
    dplyr::filter(flags_row %in% persons$flags_row[persons$included]) %>%
    dplyr::select(flags_row, set_id, trans_cond, dplyr::matches(rating_cols)) %>%
    tidyr::pivot_longer(-c(flags_row, set_id, trans_cond),
                        names_to = c("col_set", "scale_id", "aspect", "version"),
                        names_pattern = "^([A-F])([0-9]{2})_(.*)_(human|ai)$",
                        values_to = "response") %>%
    # 割り当てられたセットの尺度だけを残す(他のセットの列は空)
    dplyr::filter(col_set == set_id) %>%
    dplyr::mutate(scale_id = as.integer(scale_id), response = as.integer(response),
                  subject_id = sprintf("pro_%s_r%03d", set_id, flags_row)) %>%
    # 除外する尺度(TIVS = 11)を落とす
    dplyr::filter(!scale_id %in% exclude_scales) %>%
    dplyr::select(set = set_id, subject_id, flags_row, trans_cond, scale_id, aspect, version, response)
  # 対象になった人のうち，評定(3観点)に1つでも回答がある人に印を付ける
  answered <- long %>% dplyr::filter(!is.na(response)) %>% dplyr::distinct(subject_id) %>% dplyr::pull(subject_id)
  persons <- persons %>% dplyr::mutate(has_rating = subject_id %in% answered)
  # セットが割り当てられなかった行(同意なし・適格性で画面から外れた・途中離脱)の内訳
  unassigned <- raw %>%
    dplyr::filter(is.na(set_id)) %>%
    dplyr::mutate(consent_ok = !is.na(consent) & consent == "1",
                  any_no = dplyr::if_any(dplyr::all_of(H1_ELIGIBILITY), ~ !is.na(.x) & .x != "1"),
                  all_na = dplyr::if_all(dplyr::all_of(H1_ELIGIBILITY), ~ is.na(.x))) %>%
    dplyr::summarise(n_rows = dplyr::n(), no_consent = sum(!consent_ok),
                     eligibility_no = sum(consent_ok & any_no),
                     eligibility_unanswered = sum(consent_ok & !any_no & all_na),
                     other = sum(consent_ok & !any_no & !all_na))
  out <- list(long = long, persons = persons, unassigned = unassigned, n_rows_raw = nrow(raw))
  # 縦長の評定と人ごとの判定を返す
  return(out)
}

# 除外の人数の表(段階ごと)
h1_exclusion_table <- function(persons) {
  out <- persons %>%
    dplyr::group_by(set_id) %>%
    dplyr::summarise(
      assigned = dplyr::n(),                                     # セットが割り当てられた人
      no_consent = sum(!consent_ok),                             # 同意なし
      not_eligible = sum(consent_ok & !eligible),                # 適格性で除外
      check_wrong = sum(consent_ok & eligible & !check_ok),      # 注意チェック・AI トラップで除外
      included = sum(included),                                  # 残った人
      included_with_rating = sum(included & has_rating),         # 残った人のうち評定に回答がある人
      in_processed = sum(analysis_target == "1" & excl1 == "0"), # 共有の処理済みデータの人数(照合用)
      .groups = "drop")
  # 段階ごとの人数を返す
  return(out)
}

# 共有の long 形式(ManyScales_専門家_rating_long.csv)と値を照合する
#   共有の処理済みデータは analysis_target == 1 & excl1 == 0 の人をセット内の行順に pro_S_nnn と振っている。
#   こちらの対象のうち analysis_target == 1 の人を同じ順に並べて照合する(対の数への影響は別に数える)
h1_cross_check <- function(long, persons, rating_long_file, exclude_scales = 11) {
  # 共有の long 形式(3観点・TIVS 除く)
  ref <- readr::read_csv(rating_long_file, show_col_types = FALSE, progress = FALSE) %>%
    dplyr::filter(aspect %in% H1_ASPECTS, !scale_id %in% exclude_scales)
  # こちらの対象のうち analysis_target == 1 の人に共有側と同じ規則で番号を振る
  idmap <- persons %>%
    dplyr::filter(included, analysis_target == "1") %>%
    dplyr::arrange(set_id, flags_row) %>%
    dplyr::group_by(set_id) %>%
    dplyr::mutate(ref_id = sprintf("pro_%s_%03d", set_id, dplyr::row_number())) %>%
    dplyr::ungroup() %>%
    dplyr::select(subject_id, ref_id)
  mine <- long %>% dplyr::inner_join(idmap, by = "subject_id")
  # 値を突き合わせる(欠測どうしは一致とみなす)
  joined <- dplyr::full_join(mine %>% dplyr::select(ref_id, scale_id, aspect, version, mine = response),
                             ref %>% dplyr::select(ref_id = subject_id, scale_id, aspect, version, ref = response),
                             by = c("ref_id", "scale_id", "aspect", "version"))
  out <- tibble::tibble(
    n_rows_mine = nrow(mine), n_rows_ref = nrow(ref),
    n_persons_mine = dplyr::n_distinct(mine$ref_id), n_persons_ref = dplyr::n_distinct(ref$subject_id),
    n_mismatch = sum(!((is.na(joined$mine) & is.na(joined$ref)) |
                         (!is.na(joined$mine) & !is.na(joined$ref) & joined$mine == joined$ref))),
    # こちらだけが対象にした人(analysis_target == 0)
    n_extra_persons = sum(persons$included & persons$analysis_target != "1"),
    n_extra_rows_nonmissing = sum(!is.na(long$response[!long$subject_id %in% idmap$subject_id]))
  )
  # 照合の結果を返す
  return(out)
}


# ============================================================
# 観点ごとの整形
# ============================================================

# 1つの観点について，両版がそろう対だけを仮説3の関数に渡せる形(rating 列)にする
#   文化的妥当性の向き(2026-10-31 に訂正):
#     文化的妥当性は調査票では逆転項目(「馴染みのない慣用句…が使われている」)だが，
#     Qualtrics の時点で選択肢の値が逆向きに割り当てられている(5 = まったくそう思わない … 1 = とてもそう思う。
#     ManyScales 実装メモ・10/30 に三浦先生が確認)。つまり記録された値は既に「大きいほど文化的に妥当」なので，
#     解析では逆転をかけ直さない。
#   cv_mode(文化的妥当性だけに効く):
#     "asis"    = 記録された値のまま(主分析)
#     "reverse" = 6 − 値 に逆転(探索。10/07 の計算で主分析に使っていた扱い＝二重の逆転。差の符号が入れ替わるだけ)
h1_prepare <- function(long, aspect_name, cv_mode = c("asis", "reverse")) {
  cv_mode <- match.arg(cv_mode)
  d <- long %>% dplyr::filter(aspect == aspect_name)
  if (aspect_name == "cultural_validity" && cv_mode == "reverse") {
    d <- d %>% dplyr::mutate(response = 6L - response)
  }
  # 両版がそろう対だけを残す
  complete <- d %>%
    dplyr::group_by(subject_id, scale_id) %>%
    dplyr::filter(sum(!is.na(response)) == 2) %>%
    dplyr::ungroup()
  out <- complete %>%
    dplyr::select(set, subject_id, trans_cond, scale_id, version, rating = response)
  # 仮説3の関数と同じ列名(rating)の縦長のデータを返す
  return(out)
}


# ============================================================
# 文化的妥当性の向きの点検(記述)
# ============================================================

# 記録された値のままの文化的妥当性と他の2観点との相関(回答者×尺度×版の単位)と度数
#   記録された値が「大きいほど文化的に妥当」なら，意味的忠実性・表現の自然さと正の相関になるはず
h1_cv_check <- function(long) {
  wide <- long %>%
    tidyr::pivot_wider(id_cols = c(set, subject_id, scale_id, version),
                       names_from = aspect, values_from = response) %>%
    dplyr::filter(!is.na(cultural_validity), !is.na(semantic), !is.na(naturalness))
  # 行単位の相関(Pearson と Spearman)
  pooled <- tidyr::expand_grid(other = c("semantic", "naturalness"), method = c("pearson", "spearman")) %>%
    dplyr::mutate(n_rows = nrow(wide),
                  r = purrr::map2_dbl(other, method, function(o, m) stats::cor(wide$cultural_validity, wide[[o]], method = m)))
  # 記録された値のままの文化的妥当性の度数
  freq <- long %>% dplyr::filter(aspect == "cultural_validity", !is.na(response)) %>%
    dplyr::count(version, response)
  out <- list(pooled = pooled, freq = freq)
  # 点検の結果を返す
  return(out)
}


# ============================================================
# 尺度ごとの参考記録(差がすべて0などで計算できない尺度があっても止めない)
# ============================================================
h1_scale_bf <- function(wording_long, model_type = c("diff", "rating"), r_scale = H3_R_SCALE,
                        ci = 0.95, rope = H3_ROPE, iterations = 10000, seed = H3_SEED) {
  model_type <- match.arg(model_type)
  set.seed(seed)
  out <- wording_long %>%
    dplyr::group_split(scale_id) %>%
    purrr::map(function(g) {
      # 人ごとに版を横に並べる(対応を保つ)
      w <- tidyr::pivot_wider(g, id_cols = subject_id, names_from = version, values_from = rating)
      res <- tryCatch({
        # 差得点は対応のある t，評定は2群の t(仮説3と同じ)
        bf <- BayesFactor::ttestBF(x = w$ai, y = w$human, paired = (model_type == "diff"), rscale = r_scale)
        post <- BayesFactor::posterior(bf, iterations = iterations, progress = FALSE)
        col <- if (model_type == "diff") "mu" else "beta (x - y)"
        s <- h3_summarise_draws(as.vector(post[, "delta"]), ci = ci, rope = rope)
        tibble::tibble(bf01 = 1 / BayesFactor::extractBF(bf)$bf, diff_mean = mean(post[, col]),
                       delta_mean = s$mean, delta_hdi_low = s$hdi_low, delta_hdi_high = s$hdi_high,
                       p_rope = s$p_rope, note = NA_character_)
      }, error = function(e) {
        tibble::tibble(bf01 = NA_real_, diff_mean = NA_real_, delta_mean = NA_real_,
                       delta_hdi_low = NA_real_, delta_hdi_high = NA_real_, p_rope = NA_real_,
                       note = conditionMessage(e))
      })
      dplyr::bind_cols(tibble::tibble(scale_id = g$scale_id[1], n = nrow(w),
                                      mean_ai = mean(w$ai), mean_human = mean(w$human),
                                      mean_d = mean(w$ai - w$human)), res)
    }) %>% dplyr::bind_rows()
  # 尺度ごとの参考記録を返す
  return(out)
}
