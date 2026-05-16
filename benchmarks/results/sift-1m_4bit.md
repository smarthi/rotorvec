## sift-1m (n=1000000, queries=10000, d=128)

| Method | Bits | Recall@10 | Build (s) | Search (s) | QPS |
|---|---:|---:|---:|---:|---:|
| turbovec | 4 | 0.487 | 5.44 | 2.13 | 4702 |
| rotorvec planar2 | 4 | 0.336 | 2.39 | 3.13 | 3192 |
| rotorvec rotor3 | 4 | 0.383 | 2.53 | 3.00 | 3339 |
| rotorvec iso4 | 4 | 0.351 | 2.37 | 2.98 | 3350 |
