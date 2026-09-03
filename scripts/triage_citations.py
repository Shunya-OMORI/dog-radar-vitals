"""Chen ら (TMC 2022) を引用する 158 件を、比較対象になるかで振り分ける。

判定は 3 段階。
  段階A: ミリ波レーダ x 心臓・ECG・HRV の話題か (話題フィルタ)
  段階B: MMECG (Chen らの公開データ) を使っていそうか
  段階C: RRI / HRV 指標 (RMSSD, SDNN, pNN50, IBI) / R波検出 F1 のいずれかを報告していそうか

抄録だけの判定なので**取りこぼす**。段階B/Cに掛かったものと、A に掛かって B/C が
判定できなかったものを別々に出し、後者は本文を見て確定する。
"""
import json
import re
from collections import Counter

P = 'reports/citation_survey/citations_raw.json'
papers = json.load(open(P))

A = r'radar|mmwave|millimeter[- ]wave|rf[- ]sens|wireless sens|contactless|non[- ]contact'
A2 = r'ecg|electrocardiogra|cardiac|heart|hrv|vital sign|pulse|respirat|seismocardio'
B = r'mmecg|chen et al|public(ly)? (available )?dataset|open[- ]source dataset|\b91 (trials|recordings)\b|\b(35|11) (subjects|volunteers|participants)\b'
C_rri = r'\brri\b|rr[- ]interval|inter[- ]beat|\bibi\b|beat[- ]to[- ]beat|peak[- ]to[- ]peak interval'
C_hrv = r'\bhrv\b|rmssd|sdnn|pnn50|heart rate variability'
C_det = r'\bf1\b|f1[- ]score|precision and recall|detection rate|missed detection|\bmdr\b'
EDGE = r'edge (device|comput|deploy|inference)|embedded|microcontroller|\bmcu\b|real[- ]time on|on[- ]device|raspberry|jetson|fpga|tinyml|low[- ]power|resource[- ]constrain|lightweight|light[- ]weight|model size|parameter count|\bflops\b|quantiz|prun'

def has(p, pat):
    t = ((p.get('title') or '') + ' ' + (p.get('abstract') or '')).lower()
    return bool(re.search(pat, t))

buckets = {'A_topic': [], 'B_mmecg': [], 'C_metric': [], 'edge': [], 'no_abstract': [], 'off_topic': []}
for p in papers:
    if not p.get('abstract'):
        buckets['no_abstract'].append(p); continue
    if not (has(p, A) and has(p, A2)):
        buckets['off_topic'].append(p); continue
    buckets['A_topic'].append(p)
    if has(p, B):
        buckets['B_mmecg'].append(p)
    if has(p, C_rri) or has(p, C_hrv) or has(p, C_det):
        buckets['C_metric'].append(p)
    if has(p, EDGE):
        buckets['edge'].append(p)

print(f"引用総数: {len(papers)} 件\n")
print(f"  抄録なし(要手動確認)          : {len(buckets['no_abstract']):>3} 件")
print(f"  話題が外れる                  : {len(buckets['off_topic']):>3} 件")
print(f"  ミリ波レーダ x 心臓 (段階A)     : {len(buckets['A_topic']):>3} 件")
print(f"    うち MMECG を示唆 (段階B)    : {len(buckets['B_mmecg']):>3} 件")
print(f"    うち RRI/HRV/F1 を報告 (段階C): {len(buckets['C_metric']):>3} 件")
bc = [p for p in buckets['B_mmecg'] if p in buckets['C_metric']]
print(f"    **B かつ C (最優先の比較対象)** : {len(bc):>3} 件")
print(f"  エッジ実行に言及               : {len(buckets['edge']):>3} 件")

def dump(name, lst, n=100):
    print(f"\n{'='*70}\n{name} ({len(lst)} 件)\n{'='*70}")
    for p in sorted(lst, key=lambda x: -(x.get('year') or 0))[:n]:
        ext = p.get('externalIds') or {}
        doi = ext.get('DOI') or ext.get('ArXiv') or ''
        pdf = (p.get('openAccessPdf') or {}).get('url') or ''
        print(f"[{p.get('year')}] {(p.get('title') or '')[:95]}")
        print(f"      {p.get('venue') or '—'} | 被引用 {p.get('citationCount')} | {doi}")
        if pdf: print(f"      PDF: {pdf}")

dump('B かつ C — 最優先の比較対象', bc)
dump('エッジ実行に言及', buckets['edge'])
dump('MMECG を示唆 (B)', buckets['B_mmecg'])

json.dump({k: [{'title': p.get('title'), 'year': p.get('year'), 'venue': p.get('venue'),
                'doi': (p.get('externalIds') or {}).get('DOI'),
                'pdf': (p.get('openAccessPdf') or {}).get('url'),
                'abstract': p.get('abstract')} for p in v]
           for k, v in buckets.items()},
          open('reports/citation_survey/triage.json', 'w'), indent=2, ensure_ascii=False)
print("\n保存: reports/citation_survey/triage.json")
