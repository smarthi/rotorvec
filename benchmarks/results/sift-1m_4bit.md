## sift-1m (n=1000000, queries=10000, d=128)

| Method | Bits | Recall@10 | Build (s) | Search (s) | QPS |
|---|---:|---:|---:|---:|---:|
| turbovec | 4 | 0.487 | 5.45 | 2.15 | 4652 |
| rotorvec planar2 | 4 | 0.336 | 2.37 | 3.12 | 3208 |
| rotorvec rotor3 | 4 | 0.383 | 2.52 | 3.07 | 3258 |
| rotorvec iso4 | 4 | 0.351 | 2.40 | 3.24 | 3089 |
| rotorvec walshrotor3 | 4 | 0.496 | 3.04 | 3.11 | 3212 |
