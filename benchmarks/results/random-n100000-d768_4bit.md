## random-n100000-d768 (n=100000, queries=1000, d=768)

| Method | Bits | Recall@10 | Build (s) | Search (s) | QPS |
|---|---:|---:|---:|---:|---:|
| turbovec | 4 | 0.814 | 3.43 | 0.09 | 10715 |
| rotorvec planar2 | 4 | 0.833 | 1.56 | 16.82 | 59 |
| rotorvec rotor3 | 4 | 0.829 | 1.57 | 17.84 | 56 |
| rotorvec iso4 | 4 | 0.829 | 1.48 | 17.58 | 57 |
