# 仮説2で使う関数 ------------------------------------------------------------
# 呼び出し側で source("R/hypothesis_2_functions.R") して使う
# 関数の定義だけを置く(データの読み込み・実行・判定は呼び出し側に書く)
# データは引数で渡す
#   data: long形式の市民データ(citizen_long)
#   key : 項目×因子の対応表(R/h2_item_key_draft.csv)

# CFAの準備 ------------------------------------------------------------------
# 尺度IDと版を受け取り，wide形式のデータを作る(列名は i01, i02, ...)
make_wide <- function(data, sid, ver) {
  wide <- data |>
    dplyr::filter(scale_id == sid, version == ver) |>
    dplyr::mutate(var = sprintf("i%02d", item)) |>
    dplyr::select(subject_id, var, response) |>
    tidyr::pivot_wider(names_from = var, values_from = response) |>
    dplyr::select(-subject_id)
  return(wide)
}

# 尺度IDから lavaan のモデル式を作る(因子名は f1, f2, ... に置き換える)
make_model <- function(key, sid) {
  key_sid <- key |>
    dplyr::filter(scale_id == sid) |>
    dplyr::mutate(
      var = sprintf("i%02d", item),
      fac = paste0("f", as.integer(factor(factor, levels = unique(factor))))
    )
  model <- key_sid |>
    dplyr::group_by(fac) |>
    dplyr::summarise(rhs = paste(var, collapse = " + "), .groups = "drop") |>
    dplyr::mutate(line = paste(fac, "=~", rhs)) |>
    dplyr::pull(line) |>
    paste(collapse = "\n")
  return(model)
}

# 不適解の点検 ----------------------------------------------------------------
# lavaan の結果から不適解の中身を文字列で返す(なければ空文字)
# 対象: 1を超える標準化負荷・負の誤差分散・絶対値1以上の因子間相関・
#       潜在変数の共分散行列が正定値でない(因子が2つ以上のとき。最小固有値を書く)
describe_improper <- function(fit) {
  std <- lavaan::standardizedSolution(fit)
  lv <- lavaan::lavNames(fit, "lv")
  is_load <- std$op == "=~" & abs(std$est.std) > 1
  is_negvar <- std$op == "~~" & std$lhs == std$rhs & !(std$lhs %in% lv) & std$est.std < 0
  is_fcor <- std$op == "~~" & std$lhs != std$rhs & std$lhs %in% lv & abs(std$est.std) >= 1
  notes <- c(
    sprintf("負荷 %s→%s = %.2f", std$lhs[is_load], std$rhs[is_load], std$est.std[is_load]),
    sprintf("誤差分散 %s = %.2f", std$lhs[is_negvar], std$est.std[is_negvar]),
    sprintf("因子間相関 %s-%s = %.2f", std$lhs[is_fcor], std$rhs[is_fcor], std$est.std[is_fcor])
  )
  if (length(lv) > 1) {
    ev <- min(eigen(lavaan::lavInspect(fit, "cov.lv"), symmetric = TRUE, only.values = TRUE)$values)
    if (ev <= 0) {
      notes <- c(notes, sprintf("潜在変数の共分散行列が正定値でない(最小固有値 = %.3f)", ev))
    }
  }
  return(paste(notes, collapse = "; "))
}

# CFAの実行 ------------------------------------------------------------------
# 1尺度×1版×1推定法でCFAを当て，lavaan の結果オブジェクトを返す(失敗したら error オブジェクト)
fit_cfa <- function(data, key, sid, ver, est) {
  dat <- make_wide(data, sid, ver)
  model <- make_model(key, sid)
  ordered_vars <- if (est == "WLSMV") names(dat) else NULL
  fit <- tryCatch(
    lavaan::cfa(model, data = dat, estimator = est, ordered = ordered_vars, std.lv = TRUE),
    error = function(e) e
  )
  return(fit)
}

# 1尺度×1版×1推定法でCFAを当て，適合度を1行のtibbleで返す
fit_one <- function(data, key, sid, ver, est) {
  fit <- fit_cfa(data, key, sid, ver, est)
  if (inherits(fit, "error")) {
    return(tibble::tibble(
      scale_id = sid, version = ver, estimator = est,
      converged = FALSE, note = conditionMessage(fit)
    ))
  }
  # 頑健推定では scaled の指標を使う
  idx <- if (est == "ML") {
    c(chisq = "chisq", df = "df", cfi = "cfi", rmsea = "rmsea", srmr = "srmr")
  } else {
    c(
      chisq = "chisq.scaled", df = "df.scaled", cfi = "cfi.scaled",
      rmsea = "rmsea.scaled", srmr = "srmr"
    )
  }
  fm <- lavaan::fitMeasures(fit, idx)
  # 不適解の中身(空文字なら不適解なし)
  improper_note <- describe_improper(fit)
  # 因子間相関の最大値(1に近いと因子が弁別できていない)
  std <- lavaan::standardizedSolution(fit)
  lv <- lavaan::lavNames(fit, "lv")
  is_fcor <- std$op == "~~" & std$lhs != std$rhs & std$lhs %in% lv
  fcor <- std$est.std[is_fcor]
  fcor_pair <- paste(std$lhs[is_fcor], std$rhs[is_fcor], sep = "-")
  max_fcor <- if (length(fcor) > 0) max(abs(fcor)) else NA_real_
  max_fcor_pair <- if (length(fcor) > 0) fcor_pair[which.max(abs(fcor))] else NA_character_
  row <- tibble::tibble(
    scale_id = sid, version = ver, estimator = est,
    converged = lavaan::lavInspect(fit, "converged"),
    n = lavaan::lavInspect(fit, "nobs"),
    n_factor = length(lv),
    n_item = length(lavaan::lavNames(fit, "ov")),
    chisq = unname(fm[1]), df = unname(fm[2]), cfi = unname(fm[3]),
    rmsea = unname(fm[4]), srmr = unname(fm[5]),
    heywood = improper_note != "", improper_note = improper_note,
    max_fcor = max_fcor, max_fcor_pair = max_fcor_pair, note = ""
  )
  return(row)
}

# 適合しない箇所の診断 ----------------------------------------------------------
# 項目間のポリコリック相関行列を返す(項目がどう塊になっているかを見る)
item_cor <- function(data, sid, ver) {
  dat <- make_wide(data, sid, ver)
  r <- lavaan::lavCor(dat, ordered = names(dat))
  return(r)
}

# CFAの結果から，修正指標の上位と残差相関の上位を1つのtibbleで返す
#   type = "mi"   : 修正指標(その母数を自由にしたときの χ² の減少見込み)
#   type = "resid": 残差相関(観測相関 - モデルの再現相関)の絶対値
local_misfit <- function(fit, top = 10) {
  mi <- lavaan::modindices(fit, sort. = TRUE) |>
    tibble::as_tibble() |>
    dplyr::filter(op %in% c("=~", "~~")) |>
    dplyr::slice_head(n = top) |>
    dplyr::transmute(type = "mi", lhs, op, rhs, value = mi, sepc = sepc.all)
  res <- lavaan::lavResiduals(fit, type = "cor")$cov
  idx <- which(lower.tri(res), arr.ind = TRUE)
  resid <- tibble::tibble(
    type = "resid",
    lhs = rownames(res)[idx[, 1]],
    op = "~~",
    rhs = colnames(res)[idx[, 2]],
    value = res[idx],
    sepc = NA_real_
  ) |>
    dplyr::arrange(dplyr::desc(abs(value))) |>
    dplyr::slice_head(n = top)
  out <- dplyr::bind_rows(mi, resid)
  return(out)
}

# 探索的因子分析(WLSMV・順序扱い・geomin 回転)の負荷行列を返す
# 列は f1, f2, ... で，各項目の最大負荷の因子を top_factor に入れる
efa_check <- function(data, sid, ver, nfactors) {
  dat <- make_wide(data, sid, ver)
  efa <- lavaan::efa(
    data = dat, nfactors = nfactors, ordered = names(dat),
    estimator = "WLSMV", rotation = "geomin", output = "lavaan"
  )
  lambda <- lavaan::lavInspect(efa, "std")$lambda
  loadings <- tibble::as_tibble(unclass(lambda), rownames = "var") |>
    dplyr::rename_with(\(x) paste0("f", seq_along(x)), -var)
  loadings$top_factor <- colnames(lambda)[apply(abs(lambda), 1, which.max)] |>
    (\(x) paste0("f", match(x, colnames(lambda))))()
  return(loadings)
}

# 仮説2-2: 得点の算出と版間の比較 ---------------------------------------------
# 得点はすべて項目の平均とする
# level = "factor": 1次因子(key の factor 列)ごと
#         "upper" : 上位因子(key の factor_upper 列)ごと。factor_upper が NA の項目は使わない
#         "total" : 尺度の全項目
# どの尺度を対象にするかは，呼び出し側で key を絞って渡す
score_factors <- function(data, key, level = c("factor", "upper", "total")) {
  level <- match.arg(level)
  key_level <- key |>
    dplyr::mutate(score_name = dplyr::case_when(
      level == "factor" ~ factor,
      level == "upper" ~ factor_upper,
      level == "total" ~ "total"
    )) |>
    dplyr::filter(!is.na(score_name)) |>
    dplyr::select(scale_id, abbr, item, score_name)
  scores <- data |>
    dplyr::inner_join(key_level, by = c("scale_id", "item")) |>
    dplyr::summarise(
      n_item = dplyr::n(),
      score = mean(response),
      .by = c(subject_id, scale_id, abbr, score_name, version)
    ) |>
    dplyr::mutate(level = level, .after = abbr)
  return(scores)
}

# 得点×版ごとの記述統計(歪度・尖度は psych の既定 type = 3)
describe_scores <- function(scores) {
  desc <- scores |>
    dplyr::summarise(
      n = dplyr::n(),
      mean = mean(score),
      sd = stats::sd(score),
      median = stats::median(score),
      iqr = stats::IQR(score),
      min = min(score),
      max = max(score),
      skew = psych::skew(score),
      kurtosis = psych::kurtosi(score),
      .by = c(scale_id, abbr, level, score_name, version)
    )
  return(desc)
}

# 得点ごとに人手版とAI版を比べる(参加者内の対応あり)
# 平均の差: 対応のあるベイズ t 検定(Cauchy 事前分布，既定 rscale = √2/2)。BF10 は差あり/差なし
# 一致: ピアソンの相関係数とその95%信頼区間
compare_versions <- function(scores, rscale = sqrt(2) / 2) {
  paired <- scores |>
    dplyr::select(subject_id, scale_id, abbr, level, score_name, version, score) |>
    tidyr::pivot_wider(names_from = version, values_from = score)
  result <- paired |>
    dplyr::group_by(scale_id, abbr, level, score_name) |>
    dplyr::summarise(
      {
        bf <- BayesFactor::ttestBF(x = ai, y = human, paired = TRUE, rscale = rscale)
        ct <- stats::cor.test(ai, human, method = "pearson")
        d <- ai - human
        tibble::tibble(
          n = length(d),
          mean_human = mean(human),
          mean_ai = mean(ai),
          diff = mean(d),
          d_z = mean(d) / stats::sd(d),
          bf10 = BayesFactor::extractBF(bf)$bf,
          bf01 = 1 / bf10,
          r = unname(ct$estimate),
          r_lower = ct$conf.int[1],
          r_upper = ct$conf.int[2]
        )
      },
      .groups = "drop"
    )
  return(result)
}

# 内的整合性 ------------------------------------------------------------------
# 1尺度×1版について，得点ごとに α と ω を返す
# level は score_factors と同じ("factor"・"upper"・"total")
# α: psych::alpha の raw_alpha(素点の相関。逆転はデータの時点で済んでいるので check.keys はしない)
# ω: level = "factor" のときだけ，WLSMV(項目を順序変数として扱う)の CFA(原論文の構造)の
#    標準化負荷から (Σλ)² / ((Σλ)² + Σ(1 - λ²))。分母はモデルが示す和得点の分散
#    順序変数として推定した負荷なので，観測された和得点の信頼性ではなく，
#    潜在反応変数の上での値(順序ω)になる
#    WLSMV の CFA が推定できない・収束しない・不適解のときは(理由は omega_note)，
#    MLR(項目を連続変数として扱う)の CFA の標準化負荷から同じ式で求める(omega_method = "MLR"。表ではダガーを付ける)
#    MLR でも不適解のときは(理由は omega_note_mlr)，WLSMV の当てはめの標準化負荷からそのまま求める
#    (omega_method = "WLSMV_improper"。表では別の記号を付ける)。因子ごとの ω の式には因子間相関が入らないため
#    WLSMV でも MLR でも推定できない・収束しないときだけ，その尺度×版の ω をすべて NA にする
reliability_one <- function(data, key, sid, ver, level = c("factor", "upper", "total")) {
  level <- match.arg(level)
  dat <- make_wide(data, sid, ver)
  key_sid <- key |>
    dplyr::filter(scale_id == sid) |>
    dplyr::mutate(
      var = sprintf("i%02d", item),
      score_name = dplyr::case_when(
        level == "factor" ~ factor,
        level == "upper" ~ factor_upper,
        level == "total" ~ "total"
      )
    ) |>
    dplyr::filter(!is.na(score_name))
  groups <- split(key_sid$var, key_sid$score_name)
  alpha <- purrr::map_dbl(groups, \(v) {
    if (length(v) < 2) {
      return(NA_real_)
    }
    a <- suppressWarnings(psych::alpha(dat[, v], check.keys = FALSE, warnings = FALSE))
    return(a$total$raw_alpha)
  })
  out <- tibble::tibble(
    scale_id = sid, abbr = unique(key_sid$abbr), version = ver, level = level,
    score_name = names(groups), n_item = lengths(groups), alpha = unname(alpha),
    omega = NA_real_
  )
  if (level == "factor") {
    # make_model は因子を f1, f2, ... と出現順に名づけている
    fac_names <- unique(key_sid$factor)
    # CFA の結果から，推定できない・収束しない・不適解の理由を返す(空文字なら使える)
    check_fit <- function(fit) {
      if (inherits(fit, "error")) {
        return("推定できない")
      }
      if (!lavaan::lavInspect(fit, "converged")) {
        return("収束しない")
      }
      return(describe_improper(fit))
    }
    # 標準化負荷から因子ごとの ω を返す
    omega_from_fit <- function(fit) {
      std <- lavaan::standardizedSolution(fit)
      lam <- std[std$op == "=~", c("lhs", "est.std")]
      omega <- purrr::map_dbl(seq_along(fac_names), \(i) {
        l <- lam$est.std[lam$lhs == paste0("f", i)]
        return(sum(l)^2 / (sum(l)^2 + sum(1 - l^2)))
      })
      return(omega[match(out$score_name, fac_names)])
    }
    # まず版ごとの CFA の主分析と同じ WLSMV の当てはめを使う
    fit <- fit_cfa(data, key, sid, ver, "WLSMV")
    out$omega_note <- check_fit(fit)
    out$omega_method <- NA_character_
    out$omega_note_mlr <- ""
    if (out$omega_note[1] == "") {
      out$omega <- omega_from_fit(fit)
      out$omega_method <- "WLSMV"
    } else {
      # WLSMV が使えないときは MLR の当てはめで代える(ラフな値として示す)
      fit_mlr <- fit_cfa(data, key, sid, ver, "MLR")
      out$omega_note_mlr <- check_fit(fit_mlr)
      if (out$omega_note_mlr[1] == "") {
        out$omega <- omega_from_fit(fit_mlr)
        out$omega_method <- "MLR"
      } else if (!out$omega_note[1] %in% c("推定できない", "収束しない")) {
        # MLR も不適解なら，不適解のままの WLSMV の当てはめから求める(参考値)
        out$omega <- omega_from_fit(fit)
        out$omega_method <- "WLSMV_improper"
      }
    }
  }
  return(out)
}

# 平行分析 --------------------------------------------------------------------
# 1尺度×1版について，psych::fa.parallel(fa = "both")が示す因子数を返す
#   nfact_fa: 因子分析版(共通因子の固有値で比べる。因子を多めに出しやすい)
#   ncomp_pc: 主成分版(Horn, 1965 の元の方法)
# 相関はピアソンとポリコリックの2通り。ポリコリックは exametrika::PolychoricCorrelationMatrix で
# 求めた相関行列を人数とともに渡す(比べる乱数データは正規乱数から作られる)
# 乱数を使うので，呼び出し側で set.seed する
parallel_one <- function(data, sid, ver, n_iter = 20) {
  dat <- make_wide(data, sid, ver)
  run_pa <- function(x, n_obs = NULL) {
    # fa.parallel は結果を cat で画面に出すので capture.output で抑える
    utils::capture.output(
      pa <- suppressWarnings(suppressMessages(psych::fa.parallel(
        x, n.obs = n_obs, fa = "both", n.iter = n_iter, plot = FALSE
      )))
    )
    return(c(nfact_fa = pa$nfact, ncomp_pc = pa$ncomp))
  }
  pearson <- run_pa(dat)
  poly <- tryCatch(
    {
      r <- suppressMessages(exametrika::PolychoricCorrelationMatrix(as.matrix(dat)))
      run_pa(r, n_obs = nrow(dat))
    },
    error = function(e) c(nfact_fa = NA_real_, ncomp_pc = NA_real_)
  )
  out <- tibble::tibble(
    scale_id = sid, version = ver, n_item = ncol(dat),
    cor_type = c("pearson", "polychoric"),
    nfact_fa = c(pearson[["nfact_fa"]], poly[["nfact_fa"]]),
    ncomp_pc = c(pearson[["ncomp_pc"]], poly[["ncomp_pc"]])
  )
  return(out)
}

# 測定不変性の準備(fit_invariance と fit_partial_invariance で共通) -----------
# 1尺度の人手版とAI版を横に並べたデータを作る(1人1行，列は human_i01, ..., ai_i01, ...)
# WLSMV のときは，一方の版でしか使われていない選択肢を両版で使われている最も近い選択肢に併合する
# 返り値: wide(横長のデータ)，collapsed(併合の記録)，items(項目番号)，ids(wide の各行の人のID)
prep_long_data <- function(data, key, sid, est) {
  # --- 1. データを横に並べる(1人1行，列は human_i01, ..., ai_i01, ...) ---
  # この尺度で key に載っている項目番号
  items <- key$item[key$scale_id == sid]
  # long形式のデータから，この尺度・この項目だけを取り出す
  wide <- data |>
    dplyr::filter(scale_id == sid, item %in% items) |>
    # 列名を「版_i項目番号」にする(例: human_i01, ai_i01)
    dplyr::mutate(var = sprintf("%s_i%02d", version, item)) |>
    # 人・列名・回答だけを残す
    dplyr::select(subject_id, var, response) |>
    # 1人1行の横長にする
    tidyr::pivot_wider(names_from = var, values_from = response)
  # 人のIDは分析に使わないので外す(回答順などを後から付けられるよう，IDの並びは別に残す)
  ids <- wide$subject_id
  wide <- dplyr::select(wide, -subject_id)

  # --- 1b. (WLSMV のときだけ)両版の選択肢をそろえる ---
  # 順序データでは，同じ項目の両版で閾値の数(使われた選択肢の数)が同じでないとモデルが組めない
  # そこで項目ごとに，一方の版でしか使われていない選択肢を，両版で使われている最も近い選択肢に併合する
  # 併合した内容は collapsed_note に「i03: 1→2」の形で残す
  collapsed <- character(0)
  if (est == "WLSMV") {
    for (it in sort(unique(items))) {
      # この項目の人手版とAI版の列名
      col_h <- sprintf("human_i%02d", it)
      col_a <- sprintf("ai_i%02d", it)
      # 両版のどちらでも使われている選択肢
      common <- intersect(unique(wide[[col_h]]), unique(wide[[col_a]]))
      # 一方の版でしか使われていない選択肢
      only_one <- setdiff(union(unique(wide[[col_h]]), unique(wide[[col_a]])), common)
      for (v in sort(only_one)) {
        # 両版で使われている選択肢のうち最も近いもの(等距離なら小さい方)
        to <- common[order(abs(common - v), common)][1]
        # 両版の列でその選択肢を置き換える
        wide[[col_h]][wide[[col_h]] == v] <- to
        wide[[col_a]][wide[[col_a]] == v] <- to
        # 何をどこへ併合したかを記録する
        collapsed <- c(collapsed, sprintf("i%02d: %s→%s", it, v, to))
      }
    }
  }

  # 横長のデータ・併合の記録・項目番号をまとめて返す
  return(list(wide = wide, collapsed = collapsed, items = items, ids = ids))
}

# 1尺度の人手版とAI版の因子を並べたモデル式と，semTools に渡す両版の対応を作る
# 返り値: model(配置不変のモデル式)，long_facs(因子の両版の対応)，long_inds(項目の両版の対応)
make_long_model <- function(key, sid) {
  # --- 2. 両版の因子を1つのモデル式に書く ---
  # この尺度の項目と因子の対応(因子名は make_model と同じく出現順に f1, f2, ... とする)
  key_sid <- key |>
    dplyr::filter(scale_id == sid) |>
    dplyr::mutate(fac = paste0("f", as.integer(factor(factor, levels = unique(factor)))))
  # 1つの版について「f1_human =~ human_i01 + human_i02 ...」の行を因子ごとに作る関数
  model_one_version <- function(ver) {
    lines <- key_sid |>
      # 因子ごとに，その版の項目名を + でつなぐ
      dplyr::summarise(rhs = paste(sprintf("%s_i%02d", ver, item), collapse = " + "), .by = fac) |>
      # 「因子名_版 =~ 項目 + 項目 ...」の形にする
      dplyr::mutate(line = paste0(fac, "_", ver, " =~ ", rhs)) |>
      dplyr::pull(line)
    return(paste(lines, collapse = "\n"))
  }
  # 人手版とAI版の行をつなげて，配置不変(制約なし)のモデル式にする
  configural_model <- paste(model_one_version("human"), model_one_version("ai"), sep = "\n")

  # --- 3. semTools に「どれとどれが同じものの両版か」を教える ---
  # 因子の対応: f1 は f1_human と f1_ai，f2 は f2_human と f2_ai ...
  facs <- unique(key_sid$fac)
  long_facs <- stats::setNames(lapply(facs, \(f) paste0(f, c("_human", "_ai"))), facs)
  # 項目の対応: i01 は human_i01 と ai_i01 ...(これで同じ項目の誤差相関も自動で入る)
  item_names <- sprintf("i%02d", key_sid$item)
  long_inds <- stats::setNames(lapply(item_names, \(i) paste0(c("human_", "ai_"), i)), item_names)

  # モデル式と両版の対応をまとめて返す
  return(list(model = configural_model, long_facs = long_facs, long_inds = long_inds))
}

# 測定不変性(参加者内・縦断型) ------------------------------------------------
# 1尺度について，人手版とAI版を1つのモデルに入れ，制約を段階的に強めた各段の適合度を返す
#   WLSMV(順序扱い): 配置 → 閾値の等値 → 閾値＋負荷の等値(Wu & Estabrook, 2016 の手順)
#   MLR(連続扱い)  : 配置 → 負荷の等値 → 負荷＋切片の等値
# 同じ項目の人手版とAI版の誤差どうしには相関を置く(同じ人が同じ項目に2回答えているため)
# 対象の項目は key で決まる(IDAS-II の上位区分ごとに当てるときは，呼び出し側で key を絞って渡す)
# 判定はしない(各段の適合度・不適解の注・前の段との差を返すだけ)
fit_invariance <- function(data, key, sid, est = c("WLSMV", "MLR")) {
  # 推定法を "WLSMV" か "MLR" のどちらかに確定する
  est <- match.arg(est)

  # --- 1. データを横に並べ，(WLSMV のときだけ)両版の選択肢をそろえる(prep_long_data) ---
  prep <- prep_long_data(data, key, sid, est)
  wide <- prep$wide
  collapsed <- prep$collapsed
  items <- prep$items

  # --- 2・3. 両版のモデル式と，両版の因子・項目の対応を作る(make_long_model) ---
  lm_parts <- make_long_model(key, sid)
  configural_model <- lm_parts$model
  long_facs <- lm_parts$long_facs
  long_inds <- lm_parts$long_inds

  # --- 4. 推定法ごとに，制約の段階を決める ---
  steps <- if (est == "WLSMV") {
    # 順序データ: 閾値 → 閾値＋負荷(semTools の既定の識別 Wu.Estabrook.2016 に合わせた順)
    list(configural = character(0), thresholds = "thresholds", thresholds_loadings = c("thresholds", "loadings"))
  } else {
    # 連続データ: 負荷 → 負荷＋切片(いわゆる計量不変・尺度不変)
    list(configural = character(0), loadings = "loadings", loadings_intercepts = c("loadings", "intercepts"))
  }
  # WLSMV のときだけ全項目を順序変数として扱う
  ordered_vars <- if (est == "WLSMV") names(wide) else NULL

  # --- 5. 各段のモデルを当てる ---
  fits <- lapply(steps, \(eq) {
    # 失敗(閾値の数が両版で合わない等)してもほかの段は続けられるよう，エラーを捕まえる
    tryCatch(
      semTools::measEq.syntax(
        configural.model = configural_model, # 両版を並べたモデル式
        data = wide, # 横長のデータ
        ordered = ordered_vars, # 順序変数(MLR では NULL)
        parameterization = "delta", # 順序データの既定の母数化
        ID.fac = "std.lv", # 因子の分散を1に固定して識別(2-1 の CFA と同じ)
        ID.cat = "Wu.Estabrook.2016", # 順序データの閾値の識別の仕方
        longFacNames = long_facs, # 因子の両版の対応
        longIndNames = long_inds, # 項目の両版の対応
        long.equal = eq, # この段で両版に等値制約を置く母数
        auto = "all", # 同じ項目の両版の誤差相関をすべて入れる
        return.fit = TRUE, # 式だけでなく当てた結果(lavaan オブジェクト)を返す
        estimator = est # 推定法
      ),
      error = function(e) e
    )
  })

  # --- 6. 各段の結果を1行ずつにまとめる ---
  # 頑健推定なので scaled の指標を使う(SRMR はそのまま)
  idx <- c(chisq = "chisq.scaled", df = "df.scaled", cfi = "cfi.scaled", rmsea = "rmsea.scaled", srmr = "srmr")
  rows <- lapply(seq_along(fits), \(i) {
    fit <- fits[[i]]
    # 当てられなかった段は，適合度を NA にしてエラーの文面を残す
    # (全段が失敗しても後の ΔCFI の計算で列が足りなくならないよう，列はそろえておく)
    if (inherits(fit, "error")) {
      return(tibble::tibble(
        step = names(steps)[i], converged = FALSE,
        chisq = NA_real_, df = NA_real_, cfi = NA_real_, rmsea = NA_real_, srmr = NA_real_,
        improper_note = "", note = conditionMessage(fit)
      ))
    }
    # 収束しなかった段は適合度を出せない(fitMeasures がエラーになる)ので，NA にして注を残す
    if (!lavaan::lavInspect(fit, "converged")) {
      return(tibble::tibble(
        step = names(steps)[i], converged = FALSE,
        chisq = NA_real_, df = NA_real_, cfi = NA_real_, rmsea = NA_real_, srmr = NA_real_,
        improper_note = "", note = "収束せず"
      ))
    }
    # 適合度を取り出す
    fm <- lavaan::fitMeasures(fit, idx)
    # fitMeasures は "chisq.scaled" のような元の名前で返すので，idx の短い名前(chisq 等)に付け直す
    names(fm) <- names(idx)
    tibble::tibble(
      step = names(steps)[i], # 段の名前
      converged = lavaan::lavInspect(fit, "converged"), # 収束したか
      chisq = unname(fm["chisq"]), df = unname(fm["df"]), # カイ二乗と自由度
      cfi = unname(fm["cfi"]), rmsea = unname(fm["rmsea"]), srmr = unname(fm["srmr"]), # 適合度指標
      improper_note = describe_improper(fit), # 不適解の中身(空文字なら不適解なし)
      note = ""
    )
  })
  out <- dplyr::bind_rows(rows)

  # --- 7. 前の段との差(入れ子なので，1つ前の段と比べる) ---
  # ΔCFI = この段の CFI - 前の段の CFI(制約を強めるほど下がる)
  out$delta_cfi <- out$cfi - dplyr::lag(out$cfi)
  # 尺度化したカイ二乗差検定(WLSMV は DIFFTEST 型，MLR は Satorra-Bentler 型を lavaan が選ぶ)
  out$delta_chisq <- NA_real_
  out$delta_df <- NA_real_
  out$delta_p <- NA_real_
  for (i in seq_along(fits)[-1]) {
    # どちらかの段が当てられていない・収束していなければ差は出せない
    if (inherits(fits[[i]], "error") || inherits(fits[[i - 1]], "error")) next
    if (!lavaan::lavInspect(fits[[i]], "converged") || !lavaan::lavInspect(fits[[i - 1]], "converged")) next
    # 前の段とこの段を比べる(失敗しても止めない)
    lrt <- tryCatch(lavaan::lavTestLRT(fits[[i - 1]], fits[[i]]), error = function(e) NULL)
    if (is.null(lrt)) next
    # 結果の表の2行目が「制約を強めたモデル」との差
    out$delta_chisq[i] <- lrt[2, "Chisq diff"]
    out$delta_df[i] <- lrt[2, "Df diff"]
    out$delta_p[i] <- lrt[2, "Pr(>Chisq)"]
  }

  # 尺度ID・推定法・項目数を先頭に付け，選択肢の併合の記録を末尾に付けて返す
  out <- out |>
    dplyr::mutate(scale_id = sid, estimator = est, n_item = length(items), n = nrow(wide), .before = step) |>
    dplyr::mutate(collapsed_note = paste(collapsed, collapse = "; "))
  return(out)
}

# 部分不変性(対象項目だけを自由にする) ----------------------------------------
# 1つの下位因子について，両版の測定不変性の基準モデル(制約がいちばん強い段)と，
# 対象項目1つだけの制約を丸ごと外したモデルを当て，適合度と母数がどれだけ変わるかを返す
#   WLSMV: 基準=閾値＋負荷＋切片の等値 → その項目の閾値＋負荷を自由
#          (切片までそろえると AI版の因子平均が自由になり，アンカー項目で決まる。
#           閾値＋負荷の段では各項目の切片が自由なので，項目の位置のずれが基準の時点で吸収されてしまう)
#          (閾値だけを外すと semTools がその項目の尺度係数を1に固定するので，基準と入れ子にならない。
#           尺度係数を自由に残すと閾値＋負荷を外したモデルと同じ当てはまりになるので，1本にまとめる)
#   MLR  : 基準=負荷＋切片の等値 → その項目の切片＋負荷を自由
# 対象項目以外は自由にしない(両版をつなぐアンカーになる)
# key は呼び出し側で1つの下位因子に絞って渡す(因子が1つであることが前提)
# 判定はしない(適合度・基準との差・対象項目の母数・効果量を返すだけ)
# order(回答順の表 tibble(subject_id, first)。first は "human" か "ai")を渡すと，回答順を共変量に入れる(MIMIC)
#   x = +0.5(AI版が先)/-0.5(人手版が先)。中心化しているので，閾値・切片は回答順をならした位置を表す
#   因子への順序効果 g: f1_human ~ g*x，f1_ai ~ -g*x(2回目に答えると因子の水準がどれだけ動くか。全項目に共通)
#   対象項目への直接の順序効果 p: human_項目 ~ p*x，ai_項目 ~ -p*x(その項目だけに特有の順序効果)
#   どちらも「2回目に答える効果は人手版でもAI版でも同じ大きさ」と仮定する
#   p は項目ごとに違うので，基準モデルも対象項目ごとに当て直す
#   共変量を入れると CFI の土台(独立モデル)が変わるので，共変量なしの結果とは CFI で比べない
fit_partial_invariance <- function(data, key, sid, targets, est = c("WLSMV", "MLR"), order = NULL) {
  # 推定法を "WLSMV" か "MLR" のどちらかに確定する
  est <- match.arg(est)
  # 下位因子に絞った key でないと，効果量の計算(因子 f1 を前提)が成り立たないので止める
  fac_name <- unique(key$factor[key$scale_id == sid])
  if (length(fac_name) != 1) stop("key を1つの下位因子に絞って渡してください")

  # --- 1. データとモデル式(fit_invariance と共通) ---
  prep <- prep_long_data(data, key, sid, est)
  lm_parts <- make_long_model(key, sid)
  # WLSMV のときだけ全項目を順序変数として扱う
  ordered_vars <- if (est == "WLSMV") names(prep$wide) else NULL
  # 基準モデルで両版に等値制約を置く母数
  base_equal <- if (est == "WLSMV") c("thresholds", "loadings", "intercepts") else c("loadings", "intercepts")
  # 当てるデータ。回答順を入れるときは x を足す(順序変数ではないので ordered_vars には入れない)
  dat <- prep$wide
  if (!is.null(order)) {
    # 横長データの各行の人に，先に答えた版を付ける
    first_ver <- order$first[match(prep$ids, order$subject_id)]
    if (anyNA(first_ver)) stop("回答順の分からない人がいます")
    dat$x <- ifelse(first_ver == "ai", 0.5, -0.5)
  }

  # --- 2. 当てるモデルの一覧(基準＋対象項目ごとに1本。回答順を入れるときは基準も対象項目ごと) ---
  # long.partial には，項目を両版共通の名前(i05 など)で書く(semTools の約束)
  # add は semTools の式に足す式(回答順の効果。共変量なしなら空)
  specs <- if (is.null(order)) list(list(item = NA_integer_, model = "baseline", partial = "", add = "")) else list()
  # 項目番号は整数にそろえる(後で identical で照合するため)
  targets <- as.integer(targets)
  for (it in targets) {
    iname <- sprintf("i%02d", it)
    # 自由にする母数: WLSMV はその項目の閾値すべて，MLR は切片(これに負荷を足す)
    first <- if (est == "WLSMV") {
      # 閾値の数 = 併合後に使われている選択肢の数 - 1
      n_thr <- length(unique(stats::na.omit(prep$wide[[paste0("human_", iname)]]))) - 1
      sprintf("%s | t%d", iname, seq_len(n_thr))
    } else {
      sprintf("%s ~ 1", iname)
    }
    # 負荷も外す
    load <- sprintf("f1 =~ %s", iname)
    # 回答順の効果の式(因子への g と，この項目への p。人手版とAI版で同じ大きさ・逆向き)
    add <- if (is.null(order)) "" else paste(c(
      "f1_human ~ g*x", "f1_ai ~ g_ai*x", "g_ai == -g",
      sprintf("human_%s ~ p*x", iname), sprintf("ai_%s ~ p_ai*x", iname), "p_ai == -p"
    ), collapse = "\n")
    # 回答順を入れるときは，この項目の p を入れた基準モデルも当てる
    if (!is.null(order)) {
      specs <- c(specs, list(list(item = it, model = "baseline", partial = "", add = add)))
    }
    specs <- c(specs, list(list(item = it, model = "free", partial = c(first, load), add = add)))
  }

  # --- 3. 各モデルを当てる ---
  fits <- lapply(specs, \(sp) {
    # 失敗してもほかのモデルは続けられるよう，エラーを捕まえる
    tryCatch(
      {
        # semTools で等値制約つきのモデル式を作る(当てるのは下の lavaan)
        syn <- semTools::measEq.syntax(
          configural.model = lm_parts$model, # 両版を並べたモデル式
          data = prep$wide, # 横長のデータ(閾値の数を数えるのに使う)
          ordered = ordered_vars, # 順序変数(MLR では NULL)
          parameterization = "delta", # 順序データの既定の母数化
          ID.fac = "std.lv", # 因子の分散を1に固定して識別
          ID.cat = "Wu.Estabrook.2016", # 順序データの閾値の識別の仕方
          longFacNames = lm_parts$long_facs, # 因子の両版の対応
          longIndNames = lm_parts$long_inds, # 項目の両版の対応
          long.equal = base_equal, # 基準の等値制約
          long.partial = sp$partial, # そのうち外す制約(基準モデルは "" で何も外さない)
          auto = "all" # 同じ項目の両版の誤差相関をすべて入れる
        )
        # 回答順の効果の式を足して当てる(共変量なしなら add は空で，semTools に当てさせたのと同じ結果になる)
        lavaan::lavaan(
          paste(as.character(syn), sp$add, sep = "\n"),
          data = dat, ordered = ordered_vars, parameterization = "delta", estimator = est
        )
      },
      error = function(e) e
    )
  })
  # 当てられて収束したかどうか
  ok <- vapply(fits, \(f) !inherits(f, "error") && lavaan::lavInspect(f, "converged"), logical(1))

  # --- 4. 対象項目の母数を取り出す関数 ---
  # 1つのモデルから，ある項目の両版の負荷・閾値(または切片)と，AI版の因子の平均・分散を取り出す
  get_par <- function(fit, it) {
    pe <- lavaan::parameterEstimates(fit)
    h <- sprintf("human_i%02d", it)
    a <- sprintf("ai_i%02d", it)
    # 指定した op・lhs・rhs の推定値(rhs を省くと lhs と op だけで探す)
    val <- function(op, lhs, rhs = NULL) {
      sel <- pe$op == op & pe$lhs == lhs
      if (!is.null(rhs)) sel <- sel & pe$rhs == rhs
      return(pe$est[sel])
    }
    out <- list(
      load_human = val("=~", "f1_human", h), load_ai = val("=~", "f1_ai", a), # 負荷
      thr_human = val("|", h), thr_ai = val("|", a), # 閾値(MLR では空)
      int_human = val("~1", h), int_ai = val("~1", a), # 切片(WLSMV では両版とも 0 に固定)
      scale_ai = val("~*~", a), # AI版の尺度係数(WLSMV のみ)
      theta_human = val("~~", h, h), theta_ai = val("~~", a, a), # 誤差分散(MLR のみ)
      mean_ai = val("~1", "f1_ai"), var_ai = val("~~", "f1_ai", "f1_ai") # AI版の因子の平均と分散
    )
    return(out)
  }

  # --- 5. 期待得点の差を求める関数 ---
  # 同じ因子の値 η で両版の項目の期待得点 E[Y|η] を求め，その差を人手版の因子分布 N(0,1) の上で平均する
  # 因子の目盛りはアンカー項目で両版そろっているので，同じ η は同じ特性の高さを表す
  eta <- seq(-6, 6, length.out = 601) # 積分の格子
  w <- stats::dnorm(eta) / sum(stats::dnorm(eta)) # N(0,1) の重み
  exp_score_diff <- function(p, it) {
    if (est == "MLR") {
      # 連続扱い: E[Y|η] = 切片 + 負荷 × η なので，平均 0 の上では切片の差 + 負荷の差 × 0
      return(sum(w * ((p$int_ai - p$int_human) + (p$load_ai - p$load_human) * eta)))
    }
    # 順序扱い: 潜在反応 y* = 切片 + 負荷 × η + 誤差，y* が閾値を超えると1つ上の選択肢になる
    # (回答順を入れたときは x = 0，つまり回答順の中央での値になる)
    # delta 母数化では y* の全分散が 1/尺度係数^2 なので，誤差分散 = 1/尺度係数^2 - 負荷^2 × 因子分散
    cats <- sort(unique(stats::na.omit(prep$wide[[sprintf("human_i%02d", it)]]))) # 選択肢の値(併合後)
    ey <- function(load, thr, nu, scale, psi) {
      sd_e <- sqrt(1 / scale^2 - load^2 * psi)
      # 各閾値を超える確率(行=η，列=閾値)
      p_over <- sapply(thr, \(t) stats::pnorm((nu + load * eta - t) / sd_e))
      # 期待得点 = いちばん下の選択肢 + 閾値を超えるたびに次の選択肢との差を足す
      return(cats[1] + as.vector(p_over %*% diff(cats)))
    }
    # 人手版は因子分散 1・尺度係数 1，AI版は推定された因子分散と尺度係数(自由にした項目では 1 に固定される)
    e_h <- ey(p$load_human, p$thr_human, p$int_human, 1, 1)
    e_a <- ey(p$load_ai, p$thr_ai, p$int_ai, p$scale_ai, p$var_ai)
    return(sum(w * (e_a - e_h)))
  }

  # --- 6. 適合度を取り出す関数 ---
  # 頑健推定なので scaled の指標を使う(SRMR はそのまま)
  idx <- c(chisq = "chisq.scaled", df = "df.scaled", cfi = "cfi.scaled", rmsea = "rmsea.scaled", srmr = "srmr")
  get_fit <- function(fit) {
    fm <- lavaan::fitMeasures(fit, idx)
    names(fm) <- names(idx)
    return(as.list(fm))
  }

  # --- 7. 対象項目ごとに「基準・自由」の2行を作る ---
  rows <- list()
  for (it in targets) {
    # この項目について見る2本(基準と，この項目を自由にしたもの)
    # 共変量なしでは基準は1本を共用(item が NA)，回答順を入れたときは項目ごとの基準
    which_spec <- which(vapply(specs, \(sp) is.na(sp$item) || identical(sp$item, it), logical(1)))
    b <- which_spec[vapply(specs[which_spec], \(sp) sp$model == "baseline", logical(1))]
    base_fit <- fits[[b]]
    for (j in which_spec) {
      fit <- fits[[j]]
      row <- tibble::tibble(item = it, model = specs[[j]]$model)
      # 当てられなかった・収束しなかったモデルは，理由だけ残す
      if (!ok[j]) {
        row$note <- if (inherits(fit, "error")) conditionMessage(fit) else "収束せず"
        rows <- c(rows, list(row))
        next
      }
      # 適合度と不適解の中身
      row <- dplyr::bind_cols(row, tibble::as_tibble(get_fit(fit)))
      row$improper_note <- describe_improper(fit)
      # 基準との差(基準の行は差なし。自由にしたモデルのほうが制約が少ない)
      if (j != b && ok[b]) {
        lrt <- tryCatch(lavaan::lavTestLRT(fit, base_fit), error = function(e) NULL)
        if (!is.null(lrt)) {
          row$delta_chisq <- lrt[2, "Chisq diff"]
          row$delta_df <- lrt[2, "Df diff"]
          row$delta_p <- lrt[2, "Pr(>Chisq)"]
        }
        # ΔCFI = 自由にしたモデルの CFI - 基準の CFI(制約を外すと上がる)
        row$delta_cfi <- row$cfi - get_fit(base_fit)$cfi
      }
      # 対象項目の母数(閾値は「/」でつないだ文字列にする)
      p <- get_par(fit, it)
      row$load_human <- p$load_human
      row$load_ai <- p$load_ai
      if (est == "WLSMV") {
        row$thr_human <- paste(sprintf("%.2f", p$thr_human), collapse = "/")
        row$thr_ai <- paste(sprintf("%.2f", p$thr_ai), collapse = "/")
        # 閾値の差(AI - 人手)の平均。人手版の潜在反応の SD 単位
        row$thr_diff_mean <- mean(p$thr_ai - p$thr_human)
      } else {
        row$int_human <- p$int_human
        row$int_ai <- p$int_ai
        # 切片の差(AI - 人手)を人手版の項目の SD で割ったもの
        row$int_diff_std <- (p$int_ai - p$int_human) / stats::sd(prep$wide[[sprintf("human_i%02d", it)]], na.rm = TRUE)
      }
      # 期待得点の差(AI - 人手。素点の単位)
      row$exp_score_diff <- exp_score_diff(p, it)
      # AI版の因子の平均と分散(人手版は 0 と 1)。項目を自由にすると両版の比べ方がどれだけ動くかを見る
      row$ai_factor_mean <- p$mean_ai
      row$ai_factor_var <- p$var_ai
      # 差の検定が計算できなかったとき(不適解で尺度化の補正が NA になる等)は注を残す
      # 回答順の効果(MIMIC のときだけ): g は因子への，p はこの項目への効果(2回目 - 1回目の向きは人手版で見た値)
      if (!is.null(order)) {
        pe <- lavaan::parameterEstimates(fit)
        g <- pe[pe$label == "g" & pe$op == "~", ]
        pp <- pe[pe$label == "p" & pe$op == "~", ]
        row$order_factor <- g$est
        row$order_factor_se <- g$se
        row$order_factor_p <- g$pvalue
        row$order_item <- pp$est
        row$order_item_se <- pp$se
        row$order_item_p <- pp$pvalue
      }
      row$note <- if (j != b && is.null(row[["delta_chisq"]])) "基準との差の検定が計算できず" else ""
      rows <- c(rows, list(row))
    }
  }
  out <- dplyr::bind_rows(rows)

  # 尺度ID・下位因子・推定法・項目数・人数を先頭に付け，選択肢の併合の記録を末尾に付けて返す
  out <- out |>
    dplyr::mutate(
      scale_id = sid, factor = fac_name, estimator = est, order_adjusted = !is.null(order),
      n_item = length(prep$items), n = nrow(prep$wide), .before = item
    ) |>
    dplyr::mutate(collapsed_note = paste(prep$collapsed, collapse = "; "))
  return(out)
}
