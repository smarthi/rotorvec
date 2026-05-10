## random-n100000-d768 (n=100000, queries=1000, d=768)

| Method | Bits | Recall@10 | Build (s) | Search (s) | QPS |
|---|---:|---:|---:|---:|---:|
| turbovec | 4 | 0.814 | 3.46 | 0.09 | 10613 |
| rotorvec planar2 | 4 | 0.833 | 1.57 | 16.62 | 60 |
| rotorvec rotor3 | 4 | 0.829 | 1.56 | 17.57 | 57 |
| rotorvec iso4 | 4 | 0.829 | 1.55 | 17.62 | 57 |
