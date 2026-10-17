#### Standard pool (distortion 0.6-1.35, up to 1600 px)

| Positive images | Method | ROC-AUC | PR-AUC | Recall @1% FPR |
|---|---|---|---|---|
| phone photos of committed pages (n=76) | phash | 0.511 | 0.091 | 0.000 |
| phone photos of committed pages (n=76) | orb_raw | 0.951 | 0.653 | 0.184 |
| phone photos of committed pages (n=76) | orb_verified | 1.000 | 1.000 | 1.000 |
| phone photos of committed pages (n=76) | combined | 0.997 | 0.976 | 0.895 |
| clean digital copies (n=14) | phash | 1.000 | 0.990 | 1.000 |
| clean digital copies (n=14) | orb_raw | 0.997 | 0.902 | 0.929 |
| clean digital copies (n=14) | orb_verified | 1.000 | 1.000 | 1.000 |
| clean digital copies (n=14) | combined | 1.000 | 1.000 | 1.000 |

Correct page identified (photos): 1.00; (digital copies): 1.00
Raw ORB inliers against the WRONG paper for clean pages: median 56, max 138 -- why raw inlier counts are not used on their own.

#### Held-out stress pool (new seed, distortion 1.0-1.8, 900 px low-res forwards)

| Positive images | Method | ROC-AUC | PR-AUC | Recall @1% FPR |
|---|---|---|---|---|
| phone photos of committed pages (n=76) | phash | 0.509 | 0.091 | 0.000 |
| phone photos of committed pages (n=76) | orb_raw | 0.895 | 0.391 | 0.000 |
| phone photos of committed pages (n=76) | orb_verified | 0.961 | 0.937 | 0.934 |
| phone photos of committed pages (n=76) | combined | 0.967 | 0.869 | 0.724 |
| clean digital copies (n=17) | phash | 1.000 | 1.000 | 1.000 |
| clean digital copies (n=17) | orb_raw | 0.989 | 0.777 | 0.647 |
| clean digital copies (n=17) | orb_verified | 1.000 | 1.000 | 1.000 |
| clean digital copies (n=17) | combined | 1.000 | 1.000 | 1.000 |

Correct page identified (photos): 0.93; (digital copies): 1.00
Raw ORB inliers against the WRONG paper for clean pages: median 104, max 149 -- why raw inlier counts are not used on their own.
