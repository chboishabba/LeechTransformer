"""Independent ordered action extension of the existing mod3 task.
No model geometry imports, learned claims or reserved corpus reads.
Original Leech-LILA: Anatolii Kornienko; DASHI task extension.
"""
from collections import Counter, defaultdict
from itertools import product
import hashlib
import json

TRIADS = tuple(product((-1, 0, 1), repeat=3))
GUARDS = tuple(product(range(3), range(3), range(2), range(2)))
# a,b: translations, j: reflection, n: global negation.
# Repeated letters read the SAME assignment, never independent copies.
TRAIN_WORDS = ('a', 'b', 'j', 'n', 'ab')
HELDOUT_WORDS = ('aj', 'ja', 'ajaj', 'aajj', 'ajb', 'bja', 'abjn')
WORDS = TRAIN_WORDS + HELDOUT_WORDS


def gram(d):
    return (d[0]**2, d[1]**2, d[2]**2, d[1]*d[2], d[2]*d[0], d[0]*d[1])


def registers(word, guards, mean, residual):
    """Independent label producer: two ordered registers of signed entries.
    Translations relabel i -> i+k. Reflection relabels i -> -i and swaps
    registers. Negation negates entries. Execute tokens left to right.
    """
    a,b,j,n = guards
    p = (tuple(m+d for m,d in zip(mean,residual)),
         tuple(m-d for m,d in zip(mean,residual)))
    for token in word:
        if token in ('a','b'):
            k = a if token == 'a' else b
            p = tuple(tuple(h[(i-k)%3] for i in range(3)) for h in p)
        elif token == 'j':
            if j: p = tuple(tuple(h[(-i)%3] for i in range(3)) for h in p[::-1])
        elif token == 'n':
            if n: p = tuple(tuple(-v for v in h) for h in p)
        else: raise ValueError('unknown guard label')
    return p


def endpoint(word, guards, mean, residual):
    p,q = registers(word,guards,mean,residual)
    m = tuple((a+b)//2 for a,b in zip(p,q))
    d = tuple((a-b)//2 for a,b in zip(p,q))
    return m,d,gram(d)


def affine_word(word, guards):
    """Independent closed-form checker: index i -> (-1)^b i + k,
    with b branch swap, a global sign. Not used to produce labels.
    """
    a,b,j,n=guards;k=0;flip=0;neg=0
    for token in word:
        if token in ('a','b'): k=(k+(a if token=='a' else b))%3
        elif token=='j':
            if j:k=(-k)%3;flip^=1
        elif token=='n':neg^=n
        else:raise ValueError('unknown guard label')
    return k,flip,neg


def closed_action(word,guards,m,d):
    k,b,a=affine_word(word,guards)
    mm=[0]*3;dd=[0]*3
    for i in range(3):
        j=((-1)**b*i+k)%3
        mm[j]=(-1)**a*m[i];dd[j]=(-1)**(a+b)*d[i]
    return tuple(mm),tuple(dd),gram(dd)


def records():
    """Public finite mechanism task, not an independent language lockbox.
    Fixed initial mean e0; enumerate D and guard assignments at each word.
    """
    m=(1,0,0)
    for word,guards,d in product(WORDS,GUARDS,TRIADS):
        mm,dd,q=endpoint(word,guards,m,d)
        yield dict(word=word,guards=guards,mean=m,residual=d,mean_label=mm,
                   signed_label=dd,gram_label=q,
                   split='train' if word in TRAIN_WORDS else 'composition')


def exact_ceiling(rows, key, label):
    counts=defaultdict(Counter)
    for r in rows:counts[key(r)][r[label]]+=1
    return sum(max(c.values()) for c in counts.values()),len(rows)


def audit():
    rows=list(records());digest=hashlib.sha256()
    for r in rows:
        digest.update((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode())
        assert (r['mean_label'],r['signed_label'],r['gram_label'])==closed_action(r['word'],r['guards'],r['mean'],r['residual'])
    # Existing mod3 action: translation-only word takes e0 to e_(a+b).
    for a,b in product(range(3),repeat=2):
        m,d,q=endpoint('ab',(a,b,0,0),(1,0,0),(0,0,0))
        assert m.index(1)==(a+b)%3
    # Oracle ceilings conditioned on full word and assignments; no learned results.
    prefix=lambda r:(r['word'],r['guards'],r['mean'])
    keys={
        'mean':prefix,
        'support':lambda r:prefix(r)+(tuple(v*v for v in r['residual']),),
        'gram':lambda r:prefix(r)+(gram(r['residual']),),
        'signed':lambda r:prefix(r)+(r['residual'],),
    }
    ceilings={k:{label:exact_ceiling(rows,key,label) for label in ('mean_label','gram_label','signed_label')} for k,key in keys.items()}
    total=len(rows)
    for name,counts in {'mean':(27,2,1),'support':(27,15,8),'gram':(27,27,14),'signed':(27,27,27)}.items():
        for label,num in zip(('mean_label','gram_label','signed_label'),counts):
            assert ceilings[name][label]==(total//27*num,total)
    # Count order collisions at the SAME guards and original payload.
    reversals={}
    for w,v in (('aj','ja'),('ajaj','aajj'),('ajb','bja')):
        assert sorted(w)==sorted(v)
        ctr=Counter()
        for guards,d in product(GUARDS,TRIADS):
            p=endpoint(w,guards,(1,0,0),d);q=endpoint(v,guards,(1,0,0),d)
            for i,label in enumerate(('mean','signed','gram')):ctr[label]+=p[i]!=q[i]
        reversals[w+'/'+v]=dict(ctr,comparisons=len(GUARDS)*len(TRIADS))
    # Explicit signed and repeated-guard witnesses.
    g=(1,0,1,0);d=(1,0,0);m=(0,0,0)
    assert endpoint('aj',g,m,d)[1]==(0,0,-1)
    assert endpoint('ja',g,m,d)[1]==(0,-1,0)
    assert endpoint('ajaj',g,m,d)[1]==d
    assert endpoint('aajj',g,m,d)[1]==(0,0,1)
    # Zero vs zero-DC vs nonzero DC; no silent sign loss.
    for w,g in product(WORDS,GUARDS):
        assert endpoint(w,g,m,(0,0,0))[1]==(0,0,0)
        d=endpoint(w,g,m,(1,-1,0))[1];assert sum(d)==0 and any(d)
        d=endpoint(w,g,m,(1,1,1))[1];assert abs(sum(d))==3
    return dict(kind='x369_ordered_signed_task_v1',rows=total,
                train_rows=sum(r['split']=='train' for r in rows),
                composition_rows=sum(r['split']=='composition' for r in rows),
                train_words=TRAIN_WORDS,composition_words=HELDOUT_WORDS,
                exact_oracle_ceiling_counts=ceilings,order_separations=reversals,
                canonical_jsonl_sha256=digest.hexdigest(),
                trained_model_evaluated=False,reserved_corpus_read=False)

if __name__=='__main__':
    print(json.dumps(audit(),sort_keys=True,indent=2))
