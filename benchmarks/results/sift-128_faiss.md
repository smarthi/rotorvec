## FAISS PQ baseline — sift-128

Config: `m=128, nbits=4` — matches rotorvec's 4-bit compression rate at 512 bits/vector. FAISS trains its centroids on the data (data-aware), while turbovec/rotorvec are data-oblivious — that's the recall vs no-training trade.

| Method | Recall@10 | Build (s) | Search (s) | QPS |
|---|---:|---:|---:|---:|
| faiss IndexPQ | 0.820 | 2.81 | 75.62 | 132 |
| faiss IndexPQFastScan | 0.818 | 2.86 | 1.66 | 6042 |
