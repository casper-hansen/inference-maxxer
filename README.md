# inference-maxxer
maximum intelligence per watt

Settings: Long streaming, reasoning, tool calling requests. At batch size 1, we measure it separately where as batch size 2-8 is from the same script running these long requests.

## SGLang Kimi K2.5 8x B200

| Active Reqs | Min tok/s/req | Avg tok/s/req | Max tok/s/req |
|:-----------:|--------------:|--------------:|--------------:|
| 8 | 121.17 | 148.66 | 190.15 |
| 7 | 111.14 | 141.16 | 167.45 |
| 6 | 117.14 | 132.78 | 144.40 |
| 5 | 139.85 | 151.40 | 160.95 |
| 4 | 136.58 | 165.97 | 197.15 |
| 3 | 124.70 | 152.70 | 183.37 |
| 2 | 111.73 | 163.09 | 191.33 |
| 1 | 154.48 | 275.27 | 402.08 |
