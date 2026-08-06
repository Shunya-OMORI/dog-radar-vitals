"""非重複窓の生成と、窓系列上でのΔ(変化量)計算。

`research/CLAUDE.md`制約4「重複窓で自己相関やΔを測ってはいけない」に対応する部分を
`arousal_from_hr.py`から切り出したもの。窓の重なりを許すAPIは意図的に提供しない
(overlap引数やstride<window_secを許すオプションを作らない)。
"""

from __future__ import annotations

import numpy as np


def nonoverlapping_windows(start_s: float, end_s: float, window_sec: float) -> np.ndarray:
    """[start_s, end_s)を長さwindow_secの非重複窓に分割し、各窓の開始時刻を返す。

    端数(最後の窓がwindow_secに満たない部分)は切り捨てる
    (中途半端な短い窓でHRを計算すると分散が不当に大きくなるため)。
    """
    if window_sec <= 0:
        raise ValueError("window_secは正である必要があります")
    n_windows = int(np.floor((end_s - start_s) / window_sec))
    if n_windows <= 0:
        return np.empty(0, dtype=np.float64)
    return start_s + np.arange(n_windows, dtype=np.float64) * window_sec


def delta_over_windows(values: np.ndarray, lag: int = 1) -> np.ndarray:
    """非重複窓系列上でのΔ(変化量) = values[i+lag] - values[i] を計算する。

    重複窓ではなく、`nonoverlapping_windows`で作った非重複窓の系列に対してのみ
    使うことを想定する(そうでない系列に使うと、窓の重なりに由来する見かけの
    自己相関が再び混入する)。

    Returns
    -------
    長さ len(values) - lag の配列。delta[i] = values[i+lag] - values[i]。
    """
    values = np.asarray(values, dtype=np.float64)
    if lag <= 0:
        raise ValueError("lagは正の整数である必要があります")
    if len(values) <= lag:
        return np.empty(0, dtype=np.float64)
    return values[lag:] - values[:-lag]
