# Estimator study (v2 nights, original market)

Rule A threshold = 95th percentile of w-hat at w = 0 on calibration nights 300-599; rejection rates on nights 0-299. Pooled = TX and GA nights paired, one shared w. Selection rule (fixed in advance): highest average power over w = 10-40% subject to size and |median bias| <= 2 points.

| Variant | Market | Threshold | Reject at 0 / 10 / 20 / 30 / 40% | Avg power | Median (p10-p90) at 20% | Max median bias |
|---|---|---|---|---|---|---|
| base | TX | 38% | 9 / 10 / 13 / 22 / 73 | 29% | 20% (14-44) | 4 pts |
| base | GA | 58% | 1 / 1 / 1 / 2 / 4 | 2% | 22% (16-36) | 6 pts |
| base | pooled | 36% | 1 / 2 / 6 / 19 / 95 | 30% | 22% (18-34) | 4 pts |
| start | TX | 46% | 8 / 8 / 10 / 10 / 22 | 12% | 20% (12-46) | 6 pts |
| start | GA | 58% | 1 / 1 / 1 / 3 / 4 | 2% | 22% (16-36) | 6 pts |
| start | pooled | 38% | 1 / 2 / 4 / 14 / 86 | 26% | 22% (18-34) | 4 pts |
| start+end | TX | 46% | 9 / 10 / 10 / 10 / 21 | 13% | 22% (14-46) | 6 pts |
| start+end | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (16-36) | 6 pts |
| start+end | pooled | 36% | 1 / 3 / 6 / 16 / 95 | 30% | 22% (18-34) | 4 pts |
| start+huber | TX | 50% | 8 / 7 / 8 / 7 / 11 | 8% | 20% (12-42) | 4 pts |
| start+huber | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (18-38) | 4 pts |
| start+huber | pooled | 40% | 2 / 3 / 5 / 8 / 48 | 16% | 20% (18-32) | 4 pts |
| start+abs | TX | 52% | 8 / 8 / 7 / 6 / 9 | 8% | 20% (14-42) | 4 pts |
| start+abs | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 21% (18-36) | 4 pts |
| start+abs | pooled | 40% | 3 / 3 / 5 / 9 / 44 | 15% | 20% (18-32) | 4 pts |
| start+fineK | TX | 42% | 10 / 10 / 11 / 14 / 27 | 16% | 20% (14-46) | 6 pts |
| start+fineK | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (16-36) | 6 pts |
| start+fineK | pooled | 38% | 1 / 2 / 4 / 14 / 84 | 26% | 22% (18-34) | 4 pts |
| start+huber+fineK | TX | 50% | 7 / 7 / 8 / 6 / 11 | 8% | 20% (14-42) | 4 pts |
| start+huber+fineK | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (18-38) | 4 pts |
| start+huber+fineK | pooled | 42% | 1 / 3 / 3 / 7 / 27 | 10% | 20% (18-32) | 4 pts |
| diff3 | TX | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (4-56) | 10 pts |
| diff3 | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (6-48) | 8 pts |
| diff3 | pooled | 38% | 8 / 7 / 10 / 24 / 54 | 24% | 18% (4-38) | 6 pts |
| diff5 | TX | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 20% (2-48) | 8 pts |
| diff5 | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 20% (4-48) | 8 pts |
| diff5 | pooled | 44% | 4 / 5 / 5 / 11 / 25 | 11% | 18% (4-38) | 4 pts |
| diff10 | TX | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (2-60) | 8 pts |
| diff10 | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (4-54) | 8 pts |
| diff10 | pooled | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (4-48) | 4 pts |
| diff5+end | TX | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 22% (2-52) | 10 pts |
| diff5+end | GA | 60% | 0 / 0 / 0 / 0 / 0 | 0% | 20% (4-52) | 12 pts |
| diff5+end | pooled | 56% | 5 / 4 / 5 / 7 / 12 | 7% | 18% (4-44) | 8 pts |
