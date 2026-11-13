# Two-panel quantitative grid: paired item cosine gaps with uncertainty.
# Reuse level: structural adaptation of the existing scale-means plot.
# The reference supplies styling only; these are frequentist 95% CI, not CrI.
# Contract: 486 paired items in 21 primary scales, no additional exclusions.
# Japanese labels; numeric scale order; gray/blue/orange zero-interval coding.
# Report/slide exports: 230 x 170 mm, PNG 300 dpi, editable SVG and PDF.

build_mean_item_gap_ja <- function(repo, save = TRUE) {
  font_family <- if ("Hiragino Sans" %in% systemfonts::system_fonts()$family) "Hiragino Sans" else "Arial"
  gaps <- readr::read_csv(file.path(repo, "data/published/mpnet_item_delta.csv"),
                         show_col_types = FALSE)
  # Approved abbreviations exported from the integrated naming workbook.
  names <- readr::read_csv(file.path(repo, "data/published/scale_names_plot.csv"),
                          col_types = readr::cols(.default = readr::col_character()))
  stopifnot(nrow(names) == 22L, !anyDuplicated(names$scale_id), !anyNA(names$scale_name))
  gaps$scale_id <- sprintf("%02d", as.integer(gaps$scale_id))
  shown <- gaps[gaps$in_primary, ]
  stopifnot(nrow(gaps) == 1016L, nrow(shown) == 972L,
            length(unique(shown$scale_id)) == 21L,
            all(is.finite(shown$delta)),
            max(abs(shown$delta - (shown$human - shown$llm))) < 1e-12)
  by_scale <- shown |>
    dplyr::group_by(comparison, scale_id) |>
    dplyr::summarise(estimate = mean(delta), n_items = dplyr::n(),
                     se = stats::sd(delta) / sqrt(dplyr::n()), .groups = "drop") |>
    dplyr::mutate(ci_low = estimate - stats::qt(0.975, n_items - 1) * se,
                  ci_high = estimate + stats::qt(0.975, n_items - 1) * se,
                  interval_method = "paired-item t 95% CI") |>
    dplyr::left_join(names, by = "scale_id")
  totals <- lapply(c("back_translation", "direct_crosslingual"), function(comp) {
    part <- shown[shown$comparison == comp, ]
    stopifnot(nrow(part) == 486L)
    fit <- lme4::lmer(delta ~ 1 + (1 | scale_id), data = part, REML = TRUE)
    estimate <- unname(lme4::fixef(fit)[[1]])
    se <- sqrt(as.numeric(stats::vcov(fit)))
    data.frame(comparison = comp, scale_id = "total", scale_name = "全体",
               estimate = estimate, n_items = nrow(part), se = se,
               ci_low = estimate - 1.96 * se, ci_high = estimate + 1.96 * se,
               interval_method = "REML random-scale intercept Wald 95% CI")
  }) |>
    dplyr::bind_rows()
  stored <- readr::read_csv(file.path(repo, "data/published/specification_curve_primary.csv"),
                            show_col_types = FALSE) |>
    dplyr::filter(short_name == "mpnet", metric == "cosine")
  check <- dplyr::inner_join(totals, stored, by = "comparison", suffix = c("", "_stored"))
  stopifnot(nrow(check) == 2L, max(abs(check$estimate - check$estimate_stored)) < 1e-6,
            max(abs(check$ci_low - check$ci_low_stored)) < 2e-6,
            max(abs(check$ci_high - check$ci_high_stored)) < 2e-6)
  # Keep the exact published Python REML values after the independent R check.
  matched <- match(totals$comparison, stored$comparison)
  totals$estimate <- stored$estimate[matched]
  totals$ci_low <- stored$ci_low[matched]
  totals$ci_high <- stored$ci_high[matched]
  totals$se <- (totals$ci_high - totals$ci_low) / (2 * stats::qnorm(0.975))
  table <- dplyr::bind_rows(by_scale, totals) |>
    dplyr::mutate(
      scale_label = ifelse(scale_id == "total", "全体", paste(scale_id, scale_name)),
      direction = dplyr::case_when(ci_high < 0 ~ "LLM版が高い", ci_low > 0 ~ "人手版が高い",
                                   TRUE ~ "0を含む"),
      direction = factor(direction, levels = c("0を含む", "LLM版が高い", "人手版が高い")),
      panel = factor(comparison, levels = c("back_translation", "direct_crosslingual"),
                      labels = c("逆翻訳と原文", "日本語訳と原文（直接比較）")))
  scale_order <- by_scale |>
    dplyr::filter(comparison == "back_translation") |>
    dplyr::arrange(scale_id) |>
    dplyr::transmute(label = paste(scale_id, scale_name)) |>
    dplyr::pull(label)
  table$scale_label <- factor(table$scale_label, levels = rev(c(scale_order, "全体")))
  stopifnot(nrow(table) == 44L, all(table$ci_low <= table$estimate),
            all(table$estimate <= table$ci_high), !anyNA(table$scale_label))
  colors <- c("0を含む" = "#777777", "LLM版が高い" = "#0072B2", "人手版が高い" = "#D55E00")
  p <- ggplot2::ggplot(table, ggplot2::aes(x = estimate, y = scale_label, color = direction)) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, color = "#888888") +
    ggplot2::geom_hline(yintercept = 1.5, linewidth = 0.5, color = "#CCCCCC") +
    ggplot2::geom_linerange(ggplot2::aes(xmin = ci_low, xmax = ci_high), orientation = "y", linewidth = 0.5) +
    ggplot2::geom_point(size = 1.5) +
    ggplot2::facet_wrap(~panel, nrow = 1) +
    ggplot2::scale_color_manual(values = colors, drop = FALSE) +
    ggplot2::scale_x_continuous(breaks = scales::breaks_pretty(n = 5)) +
    ggplot2::scale_y_discrete(expand = ggplot2::expansion(add = 0.65)) +
    ggplot2::labs(x = "人手版 − LLM版(原文との項目平均類似度の差)", y = NULL, color = "95% CI") +
    ggplot2::theme_minimal(base_size = 9, base_family = font_family) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(),
                    panel.grid.major = ggplot2::element_line(color = "#EBEBEB", linewidth = 0.35),
                    strip.text = ggplot2::element_text(size = 10),
                    axis.text.y = ggplot2::element_text(size = 8),
                    axis.title.x = ggplot2::element_text(size = 10, margin = ggplot2::margin(t = 8)),
                    legend.position = "bottom", legend.text = ggplot2::element_text(size = 8),
                    panel.spacing.x = grid::unit(7, "mm"),
                    plot.margin = ggplot2::margin(8, 10, 8, 8))
  if (save) {
    dir.create(file.path(repo, "figures"), showWarnings = FALSE)
    write.csv(table, file.path(repo, "data/published/mean_item_gap_ja_ci.csv"), row.names = FALSE)
    stem <- file.path(repo, "figures/mean_item_gap_ja_ci")
    ggplot2::ggsave(paste0(stem, ".png"), p, device = ragg::agg_png,
                    width = 230, height = 170, units = "mm", dpi = 300, bg = "white")
    svglite::svglite(paste0(stem, ".svg"), width = 230 / 25.4, height = 170 / 25.4, bg = "white")
    print(p)
    grDevices::dev.off()
    grDevices::cairo_pdf(paste0(stem, ".pdf"), width = 230 / 25.4, height = 170 / 25.4,
                        family = font_family, bg = "white")
    print(p)
    grDevices::dev.off()
    file.copy(paste0(stem, ".png"), file.path(repo, "figures/r_scale_means_both.png"), overwrite = TRUE)
    one <- ggplot2::`%+%`(p, table[table$comparison == "back_translation", ])
    ggplot2::ggsave(file.path(repo, "figures/r_scale_means_primary.png"), one,
                    device = ragg::agg_png, width = 165, height = 170, units = "mm", dpi = 300, bg = "white")
  }
  list(plot = p, table = table)
}
