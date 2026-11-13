# Study translation protocol

## English

Three provider/model families were used: OpenAI `gpt-5`, Google Gemini `gemini-2.5-pro`, and Anthropic Claude `claude-opus-4-1-20250805`. Each scale was assigned one of the six ordered pairs of distinct providers. Forward translation and reconciliation used the same provider (A), and back-translation used the other (B). The local `translation/model_assignments.csv` records the scale-level assignment from the project catalog. Exact stage settings and prompts should be read from the saved materials and application logs, rather than inferred from current provider defaults.

- **LLM condition:** LLMTranslate performed forward translation (English → Japanese), back-translation (Japanese → English), and reconciliation. There were three opportunities for LLM processing.
- **Human condition:** the translation team supplied a fixed Japanese forward translation. LLMTranslate4Hitode performed back-translation and reconciliation while retaining both the original English and the supplied Japanese. There were two opportunities for LLM processing.

LLMTranslate's ordinary workflow starts with a fixed original and generates a forward translation. It cannot represent the human condition merely by supplying the human Japanese as its “original”: the English original needed for the subsequent comparison would be missing. Kunisato's Hitode modification adds the supplied forward translation as a second fixed input and skips the forward-translation call. Its Excel input has column A for the original English item and column B for the existing Japanese translation.

**Survey material selection:** respondents saw the LLM condition's initial forward Japanese translation or the translation team's initial human Japanese translation. Neither reconciled Japanese version was used. In particular, using a human translation after LLM reconciliation would undermine the intended comparison of human versus LLM translation. Back-translations are used in the H4 original-English/back-English comparison; their role does not imply that reconciled Japanese was administered.

Kunisato operated both workflows in his environment. The project reports approximate provider costs of **USD 0.19 for GPT-5**, **JPY 502 for Gemini**, and **USD 7 for Claude**. These are historical author-reported run costs, not current prices or a reproducible budget estimate.

The bundled LLMTranslate source is pinned to **0.3.0**, the version named in the supplied Hitode app and the preprint reference list. This identifies the software bundled for inspection; a complete run-by-run environment cannot be reconstructed from that citation alone. The app is preserved byte-for-byte from the author-supplied file. Study materials are supplied separately through the [OSF project](https://osf.io/hn4fr/). [Source notes](sources.md) records both sources and hashes.

## 日本語

本研究では、OpenAI（`gpt-5`）、Gemini（`gemini-2.5-pro`）、Claude（`claude-opus-4-1-20250805`）の3種類から、異なる2つを組み合わせた6パターンのいずれかを各尺度に割り当てた。順翻訳と調整は同じLLM、逆翻訳はそれと異なるLLMを用いた。

- **LLM翻訳条件：** LLMTranslateによる順翻訳 → 逆翻訳 → 調整。
- **人手翻訳条件：** 翻訳チームによる人手翻訳 → LLMTranslate4Hitodeによる逆翻訳 → 調整。

LLMTranslate4Hitodeは、国里愛彦がLLMTranslateを改変したShinyアプリケーションである。英語原文と既存の日本語訳を静的な入力として保持し、順翻訳の生成を省略して、逆翻訳と調整を行う。元のLLMTranslateの著作権表示・MIT許諾文を保持する。

**調査に使用したのは、LLM条件では順翻訳の日本語、人手条件では翻訳チームが作成した日本語であり、いずれも調整版は使用していない。** 人手条件の調整版にはLLMによる処理が加わるため、調整版を調査に用いると、人手翻訳とLLM翻訳を比較する目的が損なわれる。逆翻訳・調整の記録は、翻訳過程の資料として保持する。

両作業は国里の環境で実施した。報告された費用は、GPT-5が0.19米ドル、Geminiが502円、Claudeが約7米ドルである。現在の料金を示すものではない。
