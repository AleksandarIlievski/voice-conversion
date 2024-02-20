import os
import random
import json

# EXAMPLE dspath: '/home/jannik/datasets/DS_10283_3443/VCTK-Corpus-0.92/'
dspath = 'PATH_TO_DS_10283_3443/VCTK-Corpus-0.92/'
wavpath = dspath + 'wav48_silence_trimmed/'
exclude = ['log.txt', 's5']

with open(dspath + 'speaker-info.txt') as f:
    sgdata = f.readlines()

assert len(sgdata) - 2 == len(os.listdir(wavpath)) - len(exclude)

male = []
female = []
spks = []
for line in sgdata[1:-1]:
    g = line[10]
    spk = line[:4]
    spks.append(spk)
    if g == 'F':
        female.append(spk)
    elif g == 'M':
        male.append(spk)
    else:
        print('PROBLEM')

glists = [female, male]
print('spk len', len(spks), len(female), len(male))

utts = set()
spkutts = []
for spk in spks:
    us = list(filter(lambda x: 'mic1' in x, os.listdir(wavpath+spk+'/')))
    for u in us:
        spkutts.append(u[:8])
        utts.add(u[5:8])
# there are 503 utts and its all the numbers from 1 to 502
utts = list(utts)
print('utts len', len(utts))
print('spkutts len', len(spkutts))
U = len(utts)
N = len(spkutts)

val_ratio = 0.1
test_ratio = 0.1
train_ratio = 1 - val_ratio - test_ratio
val_N = val_ratio * N
test_N = test_ratio * N

divide_type = ('spk', 'utt')


def count_spkutts_val_test(spks, utts, spkutts):
    res = []
    for spkutt in spkutts:
        spk = spkutt[:4]
        utt = spkutt[5:8]
        if spk in spks or utt in utts:
            res.append(spkutt)
    return res

val_spks = []
val_utts = []

while len(count_spkutts_val_test(val_spks, val_utts, spkutts)) < val_N:
    div_type = divide_type[0]
    if div_type == 'spk':
        glist = glists[0]
        val_spks.append(glist.pop(random.choice(range(len(glist)))))
        glists = [glists[1], glists[0]]
    else:  # div_type == 'utt'
        val_utts.append(utts.pop(random.choice(range(len(utts))))) 
    divide_type = (divide_type[1], divide_type[0])

val_spkutts = count_spkutts_val_test(val_spks, val_utts, spkutts)
for spkutt in val_spkutts:
    spkutts.remove(spkutt)


test_spks = []
test_utts = []

while len(count_spkutts_val_test(test_spks, test_utts, spkutts)) < test_N:
    div_type = divide_type[0]
    if div_type == 'spk':
        glist = glists[0]
        test_spks.append(glist.pop(random.choice(range(len(glist)))))
        glists = [glists[1], glists[0]]
    else:  # div_type == 'utt'
        test_utts.append(utts.pop(random.choice(range(len(utts))))) 
    divide_type = (divide_type[1], divide_type[0])

test_spkutts = count_spkutts_val_test(test_spks, test_utts, spkutts)
for spkutt in test_spkutts:
    spkutts.remove(spkutt)


train_spkutts = spkutts

assert len(train_spkutts) + len(val_spkutts) + len(test_spkutts) == N
assert set(train_spkutts).intersection(set(val_spkutts)).intersection(test_spkutts) == set()

print(f"Created train set with {len(train_spkutts)} examples.")
print(f"Created val set with {len(val_spkutts)} examples.")
print(f"Created test set with {len(test_spkutts)} examples.")

out = {"train": train_spkutts, "val": val_spkutts, "test": test_spkutts}

with open("splits.json", 'w') as f:
    json.dump(out, f)

