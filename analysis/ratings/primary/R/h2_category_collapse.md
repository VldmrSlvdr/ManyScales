# 測定不変性(WLSMV)での選択肢の併合(主分析・除外基準3なし)

正本は `R/h2_category_collapse.csv`。`fit_invariance` の併合と同じ規則で作った確認用の一覧。

- 対象: 同じ項目で，一方の版でしか使われていない選択肢がある項目(TIVS を除く)
- 規則: その選択肢を，両版で使われている最も近い選択肢に併合する(等距離なら小さい方)。両版の列に同じ置き換えをかける
- 併合は測定不変性(WLSMV)のモデルだけ。2-1・2-2・内的整合性・平行分析では元の値を使う
- 合計: 11尺度・38項目・39件の併合。そのうち 29件は選んだ人が1人だけ
- 件数は保存した測定不変性の結果(`results/h2_invariance.rds` の `collapsed_note`)の併合の件数と一致する
- 除外基準3ありのデータ(参考記録)で確認したかった HEEMA の4番(人手版で選択肢2を選んだ人が0人)は，このデータでは人手版にも1人いるので併合の対象にならない。人手版の4番が全体に高い側に寄っている傾向は同じ(選択肢1と2が計5人，AI版は33人)

## IDAS-II(ID 1)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 7 | 1234 | 1,2,3,4,5 | 5 | AI | 1 | 4 | I felt a pain in my chest |
| 22 | 1234 | 1,2,3 | 4 | 人手 | 1 | 3 | I cut or burned myself on purpose |
| 64 | 1234 | 1,2,3,4,5 | 5 | AI | 1 | 4 | I felt like I had a lot of energy |
| 65 | 12345 | 1,2,3,4 | 5 | 人手 | 2 | 4 | I rearranged things so that they were in a certain order |
| 78 | 123 | 1,2,3,4 | 4 | AI | 1 | 3 | I felt like I was “on top of the world” |
| 84 | 1234 | 1,2,3,4,5 | 5 | AI | 3 | 4 | I felt anxious in small spaces |
| 88 | 1234 | 1,2,3,4,5 | 5 | AI | 1 | 4 | I had so much energy it was hard for me to sit still |
| 92 | 12345 | 1,2,3,4 | 5 | 人手 | 1 | 4 | I felt that I could do things that other people couldn’t |

## HEEMA(ID 4)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 1 | 1234567 | 2,3,4,5,6,7 | 1 | 人手 | 2 | 2 | Seeking relaxation? |
| 6 | 1234567 | 2,3,4,5,6,7 | 1 | 人手 | 2 | 2 | Seeking enjoyment? |
| 9 | 1234567 | 2,3,4,5,6,7 | 1 | 人手 | 5 | 2 | Seeking fun? |

## BTPS(ID 5)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 42 | 1234 | 1,2,3,4,5 | 5 | AI | 2 | 4 | I am the absolute best at what I do. |

## rAOT-E(ID 6)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 2 | 23456 | 1,2,3,4,5,6 | 1 | AI | 1 | 2 | People should always take into consideration evidence that g |
| 6 | 23456 | 1,2,3,4,5,6 | 1 | AI | 1 | 2 | Opinions should always be revised in response to new informa |
| 7 | 23456 | 1,2,3,4,5,6 | 1 | AI | 2 | 2 | It is possible for someone to convince me to change my mind. |

## MCS(ID 10)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 1 | 123456 | 1,2,3,4,5,6,7 | 7 | AI | 1 | 6 | I wear expensive clothes or watches so that women will be in |
| 2 | 123456 | 1,2,3,4,5,6,7 | 7 | AI | 1 | 6 | I try to make sure that I am the strongest and most muscular |
| 3 | 123456 | 1,2,3,4,5,6,7 | 7 | AI | 1 | 6 | I have a nice car because status matters to women, and I wan |
| 4 | 123456 | 1,2,3,4,5,6,7 | 7 | AI | 1 | 6 | When I'm with a woman that I'm interested in, I always try t |
| 5 | 123456 | 1,2,3,4,5 | 6 | 人手 | 1 | 5 | When I'm with a woman I'm interested in, I'll try to make ot |
| 6 | 123456 | 1,2,3,4,5,7 | 6 | 人手 | 1 | 5 | I have told a woman that I was interested in that another ma |
| 6 | 123456 | 1,2,3,4,5,7 | 7 | AI | 1 | 5 | I have told a woman that I was interested in that another ma |
| 7 | 123456 | 1,2,3,4,5 | 6 | 人手 | 1 | 5 | Sometimes I become physically aggressive with another man, s |
| 8 | 12345 | 1,2,3,4,5,6 | 6 | AI | 1 | 5 | Sometimes I'll put down another man for being cheap or drivi |
| 9 | 123456 | 1,2,3,4,5,6,7 | 7 | AI | 1 | 6 | When I'm with a woman I'm interested in, I intentionally tou |
| 11 | 123456 | 1,2,3,4,5,6,7 | 7 | AI | 1 | 6 | When I see another man talking to the woman that I'm interes |
| 13 | 12345 | 1,2,3,4,5,7 | 7 | AI | 1 | 5 | If I think another man is interested in the same woman as me |
| 16 | 12345 | 1,2,3,4,5,7 | 7 | AI | 1 | 5 | After I found out another man was interested in the same wom |

## MESx(ID 12)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 1 | 123 | 0,1,2,3 | 0 | AI | 1 | 1 | Family member |

## RMQ(ID 14)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 4 | 123456 | 2,3,4,5,6 | 1 | 人手 | 1 | 2 | I feel excited just before I am about to reach a goal. |

## MPS(ID 16)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 4 | 1234567 | 1,2,3,4,5,6 | 7 | 人手 | 1 | 6 | I seldom criticize my friends for accepting second best. |

## MEMI(ID 18)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 4 | 134567 | 1,2,3,4,5,6,7 | 2 | AI | 2 | 1 | It is delicious. |

## VAST(ID 20)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 2 | 1234 | 1,2,3,4,5 | 5 | AI | 1 | 4 | I sometimes replay my favorite scenes from gory slasher film |
| 7 | 1234 | 1,2,3,4,5 | 5 | AI | 2 | 4 | I love the YouTube clips of people fighting. |
| 8 | 1234 | 1,2,3,4,5 | 5 | AI | 1 | 4 | I enjoy physically hurting people. |
| 10 | 1234 | 1,2,3,4,5 | 5 | AI | 1 | 4 | I was purposely cruel to someone in high school. |
| 11 | 1234 | 1,2,3,4,5 | 5 | AI | 1 | 4 | I enjoy hurting my partner during sex (or pretending to). |
| 16 | 12345 | 1,2,3,4 | 5 | 人手 | 1 | 4 | I enjoy tormenting animals – especially the nasty ones. |

## GAAIS-10(ID 22)

| 項目 | 人手版で使われた選択肢 | AI版で使われた選択肢 | 片方だけの選択肢 | その版 | 選んだ人数 | 併合先 | 項目の文面 |
|---|---|---|---|---|---|---|---|
| 7 | 2345 | 1,2,3,4,5 | 1 | AI | 3 | 2 | Much of society will benefit from a future full of Artificia |

