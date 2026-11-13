// 仮説3 独立再現(探索パート): 評定モデルに尺度の評定平均と版の差の相関を入れたもの
// 天井効果で「評定が高い尺度ほど差が小さい」関係があるかを見る。fix_zero = 1 で H0
data {
  int<lower=1> N;                              // 観測の数(人×尺度×版)
  int<lower=1> J;                              // 尺度の数
  array[N] int<lower=1, upper=J> scale;        // 各観測の尺度番号
  vector[N] x;                                 // 版のコード(AI版 +1/2，人手版 −1/2)
  vector[N] y;                                 // 評定(1〜5，5 = 理解しやすい)
  real<lower=0> r_scale;                       // δ の Cauchy 事前分布の尺度
  real<lower=0> s_tau;                         // 尺度間 SD の半t事前分布の尺度
  real<lower=0> s_sigma;                       // 残差 SD の半t事前分布の尺度
  real<lower=0> lkj_eta;                       // LKJ 事前分布の形(1 = 相関について一様＝無情報)
  int<lower=0, upper=1> fix_zero;              // 1 なら H0
}
parameters {
  real alpha;                                  // 最上位の評定平均(両版の中点)
  vector[1 - fix_zero] delta;                  // 最上位の標準化した版の差(H0 では長さ0)
  vector<lower=0>[2] tau_ab;                   // 尺度間の SD(1 = 評定平均，2 = 版の差)
  cholesky_factor_corr[2] L_ab;                // 尺度の評定平均と版の差の相関のコレスキー因子
  matrix[2, J] z_ab;                           // 尺度の効果(非中心化のための標準正規)
  real<lower=0> sigma;                         // 残差 SD
}
transformed parameters {
  // 最上位の版の差(評定の単位)。H0 では0のまま
  real beta = 0;
  // H1 では標準化した差 δ に残差 SD を掛けて評定の単位に戻す
  if (!fix_zero) beta = delta[1] * sigma;
  // 相関つきの尺度の効果(J × 2。1列目 = 評定平均のずれ，2列目 = 版の差のずれ)
  matrix[J, 2] u_ab = (diag_pre_multiply(tau_ab, L_ab) * z_ab)';
  // 尺度ごとの評定平均
  vector[J] alpha_scale = alpha + u_ab[, 1];
  // 尺度ごとの版の差
  vector[J] beta_scale = beta + u_ab[, 2];
}
model {
  // δ の事前分布(H1 のみ・定数込み)
  if (!fix_zero) target += cauchy_lpdf(delta | 0, r_scale);
  // 評定平均の事前分布
  target += normal_lpdf(alpha | 3, 2);
  // 尺度間 SD の事前分布(半t。主分析と同じ尺度)
  target += student_t_lpdf(tau_ab | 3, 0, s_tau) + num_elements(tau_ab) * log2();  // 半t: 0以上に折り返すので密度を2倍(要素ごとに log 2)
  // 相関の事前分布(LKJ)
  target += lkj_corr_cholesky_lpdf(L_ab | lkj_eta);
  // 尺度の効果の標準正規
  target += std_normal_lpdf(to_vector(z_ab));
  // 残差 SD の事前分布(半t)
  target += student_t_lpdf(sigma | 3, 0, s_sigma) + log2();  // 半t: 0以上に折り返すので密度を2倍
  // 尤度
  target += normal_lpdf(y | alpha_scale[scale] + beta_scale[scale] .* x, sigma);
}
generated quantities {
  // 尺度の評定平均と版の差の相関
  real rho_ab = multiply_lower_tri_self_transpose(L_ab)[1, 2];
  // 新しい尺度の版の差の予測(相関があっても周辺は正規)
  real beta_new = normal_rng(beta, tau_ab[2]);
}
