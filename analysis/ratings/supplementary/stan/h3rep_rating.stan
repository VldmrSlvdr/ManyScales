// 仮説3 独立再現: 評定モデル(版の効果)
// fix_zero = 1 で H0(最上位の版の差を0に固定)
data {
  int<lower=1> N;                              // 観測の数(人×尺度×版)
  int<lower=1> J;                              // 尺度の数
  array[N] int<lower=1, upper=J> scale;        // 各観測の尺度番号
  vector[N] x;                                 // 版のコード(AI版 +1/2，人手版 −1/2)
  vector[N] y;                                 // 評定(1〜5，5 = 理解しやすい)
  real<lower=0> r_scale;                       // δ の Cauchy 事前分布の尺度
  real<lower=0> s_tau;                         // 尺度間 SD の半t事前分布の尺度(主分析は 0.5)
  real<lower=0> s_sigma;                       // 残差 SD の半t事前分布の尺度(主分析は 1)
  int<lower=0, upper=1> fix_zero;              // 1 なら H0
}
parameters {
  real alpha;                                  // 最上位の評定平均(両版の中点)
  real<lower=0> tau_alpha;                     // 尺度間の評定平均の SD
  vector[J] z_alpha;                           // 尺度の評定平均(非中心化)
  vector[1 - fix_zero] delta;                  // 最上位の標準化した版の差(H0 では長さ0)
  real<lower=0> tau_beta;                      // 尺度間の版の差の SD
  vector[J] z_beta;                            // 尺度ごとの版の差(非中心化)
  real<lower=0> sigma;                         // 残差 SD(回答者間のばらつきを含む = 2群の d 相当)
}
transformed parameters {
  // 最上位の版の差(評定の単位)。H0 では0のまま
  real beta = 0;
  // H1 では標準化した差 δ に残差 SD を掛けて評定の単位に戻す
  if (!fix_zero) beta = delta[1] * sigma;
  // 尺度ごとの評定平均
  vector[J] alpha_scale = alpha + tau_alpha * z_alpha;
  // 尺度ごとの版の差
  vector[J] beta_scale = beta + tau_beta * z_beta;
}
model {
  // δ の事前分布(H1 のみ・定数込み)
  if (!fix_zero) target += cauchy_lpdf(delta | 0, r_scale);
  // 評定平均の事前分布(5件法の中点3を中心に広く)
  target += normal_lpdf(alpha | 3, 2);
  // 尺度間 SD の事前分布(半t)
  target += student_t_lpdf(tau_alpha | 3, 0, s_tau) + log2();  // 半t: 0以上に折り返すので密度を2倍
  target += student_t_lpdf(tau_beta | 3, 0, s_tau) + log2();  // 半t: 0以上に折り返すので密度を2倍
  // 残差 SD の事前分布(半t)
  target += student_t_lpdf(sigma | 3, 0, s_sigma) + log2();  // 半t: 0以上に折り返すので密度を2倍
  // 尺度の効果の標準正規
  target += std_normal_lpdf(z_alpha);
  target += std_normal_lpdf(z_beta);
  // 尤度
  target += normal_lpdf(y | alpha_scale[scale] + beta_scale[scale] .* x, sigma);
}
generated quantities {
  // 新しい尺度の版の差の予測
  real beta_new = normal_rng(beta, tau_beta);
  // 最上位の版ごとの評定平均(図用)
  real mean_ai = alpha + beta / 2;
  real mean_human = alpha - beta / 2;
}
