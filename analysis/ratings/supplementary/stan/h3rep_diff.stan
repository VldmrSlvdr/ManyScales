// 仮説3 独立再現: 差得点モデル(AI版 − 人手版)
// fix_zero = 1 で H0(最上位の差を0に固定)。use_person・use_order は探索パート用
data {
  int<lower=1> N;                              // 観測の数(人×尺度の対の数)
  int<lower=1> J;                              // 尺度の数
  array[N] int<lower=1, upper=J> scale;        // 各観測の尺度番号(1..J に振り直したもの)
  vector[N] d;                                 // 差得点(AI版の評定 − 人手版の評定)
  real<lower=0> r_scale;                       // δ の Cauchy 事前分布の尺度(主分析は sqrt(2)/2)
  real<lower=0> s_tau;                         // 尺度間 SD の半t事前分布の尺度(主分析は 0.5)
  real<lower=0> s_sigma;                       // 残差 SD の半t事前分布の尺度(主分析は 1)
  int<lower=0, upper=1> fix_zero;              // 1 なら H0
  int<lower=0, upper=1> use_person;            // 1 なら回答者の効果を入れる(探索パートのみ)
  int<lower=1> I;                              // 回答者の数(use_person = 0 でも渡す)
  array[N] int<lower=1, upper=I> person;       // 各観測の回答者番号
  int<lower=0, upper=1> use_order;             // 1 なら先に答えた版の項を入れる(探索パートのみ)
  vector[N] ai_first;                          // AI版を先に答えたら +1/2，人手版が先なら −1/2
}
parameters {
  vector[1 - fix_zero] delta;                  // 最上位の標準化した差(H0 では長さ0)
  real<lower=0> tau;                           // 尺度間の SD
  vector[J] z_scale;                           // 尺度の効果(非中心化のための標準正規)
  real<lower=0> sigma;                         // 尺度内の差得点の残差 SD
  vector<lower=0>[use_person] tau_person;      // 回答者間の SD(探索パートのみ)
  vector[use_person ? I : 0] z_person;         // 回答者の効果(非中心化)
  vector[use_order] gamma;                     // 先に答えた版の効果(探索パートのみ)
}
transformed parameters {
  // 標準化の分母(主分析では尺度内の差得点の SD = d_z 相当)
  real sd_unit = sigma;
  // 最上位の差(評定の単位)。H0 では0のまま
  real mu = 0;
  // 回答者の効果を入れたときは，その分散も分母に足す(探索パートのみ)
  if (use_person) sd_unit = sqrt(square(sigma) + square(tau_person[1]));
  // H1 では標準化した差 δ に分母を掛けて評定の単位に戻す
  if (!fix_zero) mu = delta[1] * sd_unit;
  // 尺度ごとの差 μ_j
  vector[J] mu_scale = mu + tau * z_scale;
}
model {
  // 観測ごとの期待値を尺度の差から作る
  vector[N] eta = mu_scale[scale];
  // 回答者の効果を足す(探索パートのみ)
  if (use_person) eta += tau_person[1] * z_person[person];
  // 先に答えた版の効果を足す(探索パートのみ)
  if (use_order) eta += gamma[1] * ai_first;
  // δ の事前分布(H1 のみ)。BF のために ~ でなく target += で定数込みにする
  if (!fix_zero) target += cauchy_lpdf(delta | 0, r_scale);
  // 尺度間 SD の事前分布(半t。H0 と H1 で共通なので正規化定数は BF で打ち消し合う)
  target += student_t_lpdf(tau | 3, 0, s_tau) + log2();  // 半t: 0以上に折り返すので密度を2倍
  // 残差 SD の事前分布(半t)
  target += student_t_lpdf(sigma | 3, 0, s_sigma) + log2();  // 半t: 0以上に折り返すので密度を2倍
  // 尺度の効果の標準正規
  target += std_normal_lpdf(z_scale);
  // 回答者の効果の事前分布(探索パートのみ)
  if (use_person) {
    target += student_t_lpdf(tau_person | 3, 0, s_tau) + num_elements(tau_person) * log2();  // 半t: 0以上に折り返すので密度を2倍(要素ごとに log 2)
    target += std_normal_lpdf(z_person);
  }
  // 順序の効果の事前分布(探索パートのみ)
  if (use_order) target += normal_lpdf(gamma | 0, 1);
  // 尤度
  target += normal_lpdf(d | eta, sigma);
}
generated quantities {
  // 新しい尺度の差の予測(フォレストプロットの最下段に使う)
  real mu_new = normal_rng(mu, tau);
}
