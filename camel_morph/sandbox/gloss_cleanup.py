import re

problematic_glosses = {}

def clean_gloss(gloss):
    for x in [' ', '#', ',']:
        if x in gloss:
            problematic_glosses.setdefault(x, []).append(gloss)
    
    gloss_ = gloss.replace(',', '')
    gloss_ = re.split(r'#+|;+', gloss_)
    gloss_ = set(g for g in map(str.strip, gloss_) if g)
    if len(gloss_) > 1 and 'TBA' in gloss_:
        gloss_ = gloss_ - {'TBA'}
    gloss_ = sorted(gloss_)
    gloss_ = ';'.join(gloss_)

    assert re.search(r'[#, ]', gloss_) is None
    return gloss_


with open('scratch_files/gloss_cleanup.tsv') as f:
    glosses = [gloss.strip() for gloss in f.readlines()]

with open('scratch_files/gloss_cleanup_output.tsv', 'w') as f:
    for gloss in glosses:
        gloss_ = clean_gloss(gloss)
        print(gloss_, file=f)

pass