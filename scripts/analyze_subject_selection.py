"""被験者ごとの難易度と、「テスト被験者の選び方」が報告値をどれだけ動かすかを出す。

背景 (2026-08-25、田中先生から別研究へのコメント):
    「困難な被験者をあえて選択して問題を難しくする必要はない」
    「被験者の数が有限なら、どの被験者だと精度が出やすいの? という問いに応えられるように」

被験者ごとにレーダ反射の性質と体動が大きく異なるので、テスト被験者の選択は報告値を
大きく動かす。ここでは次を出す。

  (1) 被験者ごとの成績 (共通しきい値と、その被験者の最良しきい値の両方)
  (2) 11 名から 3 名をテストに選ぶ全 165 通りで、報告値がどこまで動くか
  (3) 成績を説明できる入力側の量はあるか (順位相関)

使い方:
    python scripts/analyze_subject_selection.py --sweep reports/height_sweep_masked_combined.json
"""
import argparse
import itertools
import json
import os

import numpy as np

DIAG = 'reports/subject_difficulty_diagnosis.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True)
    ap.add_argument('--common_h', default='0.3')
    ap.add_argument('--label', default='302 (MAD正規化 + 正例重み)')
    ap.add_argument('--out', default='reports/subject_selection_analysis.json')
    args = ap.parse_args()

    sweep = json.load(open(args.sweep))
    subs = sorted(sweep.keys(), key=int)
    hs = sorted(sweep[subs[0]].keys(), key=float)
    ch = args.common_h if args.common_h in sweep[subs[0]] else hs[0]

    print(f"=== 被験者ごとの成績 — {args.label} ===")
    print(f"共通しきい値 h={ch}。「最良h」はその被験者だけを見て選んだ場合（本来は使えない）\n")
    print(f"{'被験者':>5} {'F1':>7} {'RR間隔':>8} {'RMSSD':>8} {'一定RR':>8} {'最良F1':>8} {'(h)':>6} "
          f"{'最良RMSSD':>10} {'(h)':>6}")
    rows = []
    for s in subs:
        d = sweep[s][ch]
        f1s = {h: sweep[s][h]['f1'] for h in hs}
        rms = {h: sweep[s][h]['rmssd'] for h in hs}
        bf_h = max(f1s, key=lambda h: f1s[h])
        br_h = min(rms, key=lambda h: rms[h])
        print(f"{s:>5} {d['f1']:7.3f} {d['rr']:8.2f} {d['rmssd']:8.2f} {d['triv_rmssd']:8.2f} "
              f"{f1s[bf_h]:8.3f} {bf_h:>6} {rms[br_h]:10.2f} {br_h:>6}")
        rows.append(dict(subject=int(s), f1=d['f1'], rr=d['rr'], rmssd=d['rmssd'],
                         triv_rmssd=d['triv_rmssd'], best_f1=f1s[bf_h], best_f1_h=bf_h,
                         best_rmssd=rms[br_h], best_rmssd_h=br_h))

    f1_arr = np.array([r['f1'] for r in rows])
    order = np.argsort(-f1_arr)
    print(f"\n  F1 が高い順: " + ' > '.join(str(rows[i]['subject']) for i in order))
    print(f"  F1 の範囲: {f1_arr.min():.3f} (被験者 {rows[int(np.argmin(f1_arr))]['subject']}) "
          f"〜 {f1_arr.max():.3f} (被験者 {rows[int(np.argmax(f1_arr))]['subject']})")

    # (2) テスト3名の選び方で報告値がどこまで動くか
    print(f"\n=== 11 名から 3 名をテストに選ぶ全 {len(list(itertools.combinations(range(11),3)))} 通り ===")
    combos = []
    for c in itertools.combinations(range(len(subs)), 3):
        f1 = float(np.mean([rows[i]['f1'] for i in c]))
        rr = float(np.mean([rows[i]['rr'] for i in c]))
        rm = float(np.mean([rows[i]['rmssd'] for i in c]))
        combos.append((f1, rr, rm, tuple(rows[i]['subject'] for i in c)))
    combos.sort(key=lambda x: -x[0])
    print(f"  {'':4} {'F1':>7} {'RR間隔':>8} {'RMSSD':>8}  テスト被験者")
    print(f"  最良 {combos[0][0]:7.3f} {combos[0][1]:8.2f} {combos[0][2]:8.2f}  {combos[0][3]}")
    mid = combos[len(combos)//2]
    print(f"  中央 {mid[0]:7.3f} {mid[1]:8.2f} {mid[2]:8.2f}  {mid[3]}")
    print(f"  最悪 {combos[-1][0]:7.3f} {combos[-1][1]:8.2f} {combos[-1][2]:8.2f}  {combos[-1][3]}")
    allf1 = np.array([c[0] for c in combos])
    print(f"\n  **F1 は選び方だけで {allf1.min():.3f} 〜 {allf1.max():.3f} "
          f"({allf1.max()-allf1.min():.3f} の幅) 動く。** 11分割の平均は {f1_arr.mean():.3f}")
    print(f"  RMSSD 誤差は {min(c[2] for c in combos):.2f} 〜 {max(c[2] for c in combos):.2f} ms")

    # (3) 入力側の量で説明できるか
    if os.path.exists(DIAG):
        diag = {int(d['subject']): d for d in json.load(open(DIAG))} \
            if isinstance(json.load(open(DIAG)), list) else None
        if diag:
            print("\n=== 成績を入力側の量で説明できるか (Spearman 順位相関, n=11) ===")
            def rank(v):
                v = np.asarray(v, float); o = v.argsort(); r = np.empty_like(o, float)
                r[o] = np.arange(len(v)); return r
            keys = [k for k in diag[rows[0]['subject']] if k not in ('subject',)
                    and isinstance(diag[rows[0]['subject']][k], (int, float))]
            for k in keys:
                try:
                    x = np.array([diag[r['subject']][k] for r in rows], float)
                except (KeyError, TypeError):
                    continue
                if np.allclose(x, x[0]):
                    continue
                rho = float(np.corrcoef(rank(x), rank(f1_arr))[0, 1])
                print(f"  {k:16s} vs F1 : rho = {rho:+.3f}")
            print("\n  n=11 なので |rho| が 0.6 程度でも偶然の範囲に入りうる。")

    json.dump(dict(per_subject=rows, common_h=ch,
                   combo_best=combos[0], combo_worst=combos[-1],
                   combo_f1_min=float(allf1.min()), combo_f1_max=float(allf1.max())),
              open(args.out, 'w'), indent=2, ensure_ascii=False)
    print(f"\n保存: {args.out}")


if __name__ == '__main__':
    main()
