import json

with open('splits.json') as f:
    data = json.load(f)

train = data['train']
val = data['val']
test = data['test']

print('')

train_spks = set([item[:4] for item in train])
val_spks = set([item[:4] for item in val])
print('num train spks:', len(train_spks))
print('num val spks:', len(val_spks))
print('same spks', len(train_spks.intersection(val_spks)))

train_utts = set([item[5:] for item in train])
val_utts = set([item[5:] for item in val])
print('num train utts:', len(train_utts))
print('num val utts:', len(val_utts))
print('same utts', len(train_utts.intersection(val_utts)))

val_uu = [item for item in val if item[:4] not in train_spks and item[5:] not in train_utts]
print('val uu', len(val_uu))
test_uu = [item for item in test if item[:4] not in train_spks and item[5:] not in train_utts]
print('test uu', len(test_uu))
test_uu = [item for item in test if item[:4] not in train_spks and item[5:] not in train_utts]
print('test uu', len(test_uu))

print(train[:10])

print(set(train).intersection(set(val)))

data['val_uu'] = val_uu
data['test_uu'] = test_uu

with open('splits_ext.json', 'w') as f:
    json.dump(data, f)