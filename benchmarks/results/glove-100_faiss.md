## FAISS PQ baseline — glove-100

Config: `m=100, nbits=4` — matches rotorvec's 4-bit compression rate at 400 bits/vector. FAISS trains its centroids on the data (data-aware), while turbovec/rotorvec are data-oblivious — that's the recall vs no-training trade.

| Method | Recall@10 | Build (s) | Search (s) | QPS |
|---|---:|---:|---:|---:|
| faiss IndexPQ | 0.810 | 2.67 | 75.32 | 133 |
| faiss IndexPQFastScan | 0.806 | 2.94 | 1.68 | 5940 |
