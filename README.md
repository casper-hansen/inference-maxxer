# inference-maxxer
maximum intelligence per watt

Settings: Long streaming, reasoning, tool calling requests. At batch size 1, we measure it separately where as batch size 2-8 is from the same script running these long requests.

## vLLM Kimi K2.5 8x H200

| Concurrency | Min tok/s/req | Avg tok/s/req | Max tok/s/req | Time to first token |
|:-----------:|--------------:|--------------:|--------------:|--------------------:|
| 8 | 68.73 | 128.40 | 253.72 | 1541 ms |
| 7 | 69.26 | 121.03 | 315.88 | 2668 ms |
| 6 | 77.16 | 138.79 | 285.91 | 2206 ms |
| 5 | 82.64 | 142.95 | 291.95 | 1720 ms |
| 4 | 92.27 | 165.80 | 316.62 | 1077 ms |
| 3 | 91.03 | 155.85 | 316.48 | 1804 ms |
| 2 | 103.48 | 168.88 | 319.58 | 1590 ms |
| 1 | 142.53 | 233.74 | 352.26 | 1569 ms |
