"""Nine-level recovery diagnostics, including exact tied-score cut enumeration."""
from __future__ import annotations
import math

def cut_predictions(margins):
    values=[float(x) for x in margins]
    if len(values)!=9 or not all(math.isfinite(x) for x in values):
        raise ValueError('Exactly nine finite margins required')
    return [[x>=cut for x in values] for cut in sorted(set(values))]+[[False]*9]

def curve_diagnostics(margins, j_star):
    if not 1<=j_star<=9: raise ValueError('j_star must be z+1, in 1..9')
    cuts=cut_predictions(margins)
    truth=[i<j_star for i in range(9)]
    prediction=[x>=0 for x in margins]
    first=next((i for i,v in enumerate(prediction) if not v),9)
    errors=sum(a!=b for a,b in zip(prediction,truth))
    trr=None if j_star==9 else margins[j_star]<min(margins[:j_star])
    fsrr=None if j_star==9 else min(margins[:j_star])>max(margins[j_star:])
    return dict(z=j_star-1,j_star=j_star,accuracy=1-errors/9,
                strict_accuracy=sum(prediction[i]==truth[i] for i in range(1,9))/8,
                errors=errors,trr=trr,fsrr=fsrr,predicted_first_fail=first,tce=abs(first-j_star),
                oracle_min_errors=min(sum(a!=b for a,b in zip(p,truth)) for p in cuts),
                trr_gap=None if j_star==9 else min(margins[:j_star])-margins[j_star],
                fsrr_gap=None if j_star==9 else min(margins[:j_star])-max(margins[j_star:]),
                adjacent_increases=[margins[i+1]-margins[i] for i in range(8) if margins[i+1]>margins[i]])
