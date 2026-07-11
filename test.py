
# import mne, numpy as np
# from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy
# from pyprep.find_noisy_channels import NoisyChannels
# print('Starting test...')
# edf = "/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data/v3.0.0/01_no_epilepsy/aaaaapsj/s001_2013/01_tcp_ar/aaaaapsj_s001_t002.edf"            # the recording you tested
# raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
# TUHEEGEpilepsy._rename_channels(raw)
# raw.pick([c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"])
# TUHEEGEpilepsy._set_montage(raw)
# det = raw.copy().filter(1.0, None, verbose="error")
# det.notch_filter(60, verbose="error")

# d = det.get_data()
# ptp_uv = (d.max(1) - d.min(1)) * 1e6
# print(f"scale: median ptp = {np.median(ptp_uv):.1f} uV  (min {ptp_uv.min():.1f}, max {ptp_uv.max():.1f})")
# pos = (det.get_montage().get_positions() or {}).get("ch_pos", {})
# n_pos = sum(1 for c in det.ch_names if c in pos and np.all(np.isfinite(pos[c])))
# print(f"positions: {n_pos}/{len(det.ch_names)} channels have montage coords")

# nc = NoisyChannels(det, do_detrend=False); nc.find_all_bads(ransac=True)
# for a in ("bad_by_nan","bad_by_flat","bad_by_deviation","bad_by_hf_noise",
#           "bad_by_correlation","bad_by_SNR","bad_by_dropout","bad_by_ransac"):
#     print(f"  {a}: {getattr(nc, a, '?')}")

import json, glob, numpy as np
funds, flagged = [], 0
for p in glob.glob("data/v3.0.0/**/*-bads.json", recursive=True):
    pa = (json.load(open(p)) or {}).get("periodic_artifact") or {}
    if pa.get("flagged"):
        flagged += 1
        funds.append(pa.get("fundamental_hz"))
f = np.array([x for x in funds if x is not None])
print(f"flagged={flagged}  with_fundamental={len(f)}  none={sum(x is None for x in funds)}")
for lo, hi in [(0.5,0.7),(0.7,1.2),(1.2,2.0),(2.0,3.5),(3.5,6.01),(6.01,10.0),(10.0,20.0),(20.0,40.0),(40.0,80.0),(80.0,160.0)]:
    print(f"  [{lo},{hi}) Hz: {int(((f>=lo)&(f<hi)).sum())}")