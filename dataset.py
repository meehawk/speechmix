import os
import numpy as np
import random
import chainer
from sklearn.model_selection import train_test_split

import utils as U


class SoundDataset_Train(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, labels, opt, train=True):
        self.base = chainer.datasets.TupleDataset(sounds, labels)
        self.opt = opt
        self.train = train
        self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)//2

    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def get_example(self, i):
        sound1 = None
        sound2 = None
        label1 = None
        label2 = None
        if self.mix:  # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            # r = np.array(random.random())
            # sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label1 = eye[label1]
            label2 = eye[label2]
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
            # label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

        else:  # Training phase of standard learning or testing phase
            sound, label = self.base[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

        if self.train and self.opt.strongAugment:
            sound1 = U.random_gain(6)(sound1).astype(np.float32)
            sound2 = U.random_gain(6)(sound1).astype(np.float32)
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
        return sound, label


class SoundDataset_Val(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, labels, opt, train=True):
        self.base = chainer.datasets.TupleDataset(sounds, labels)
        self.opt = opt
        self.train = train
        self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)

    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def get_example(self, i):
        if self.mix:  # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            # r = np.array(random.random())
            # sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label1 = eye[label1]
            label2 = eye[label2]
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
            # label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

        else:  # Training phase of standard learning or testing phase
            sound, label = self.base[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

        if self.train and self.opt.strongAugment:
            sound1 = U.random_gain(6)(sound1).astype(np.float32)
            sound2 = U.random_gain(6)(sound1).astype(np.float32)
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
        return sound, label


def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)), allow_pickle = True)

    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []

    if opt.dataset in ['urdu', 'savee', 'emodb', 'emovo', 'shemo']:
        train_sounds, val_sounds, train_labels, val_labels = train_test_split(
            dataset['sounds'],
            dataset['labels'],
            test_size=0.2,
            shuffle=True,
            stratify=dataset['labels'],
            random_state=42,
        )

        if len(train_sounds) % 2 != 0:
            train_sounds = train_sounds[:-1]
            train_labels = train_labels[:-1]
    else:
        for i in range(1, opt.nFolds + 1):
            sounds = dataset[f'fold{i}'].item()['sounds']
            labels = dataset[f'fold{i}'].item()['labels']
            if i == split:
                val_sounds.extend(sounds)
                val_labels.extend(labels)
            else:
                train_sounds.extend(sounds)
                train_labels.extend(labels)

    # Iterator setup
    train_data = SoundDataset_Train(train_sounds, train_labels, opt, train=True)
    val_data = SoundDataset_Val(val_sounds, val_labels, opt, train=False)
    train_iter = chainer.iterators.MultiprocessIterator(train_data, opt.batchSize, repeat=False)
    val_iter = chainer.iterators.SerialIterator(val_data, 1, repeat=False, shuffle=False)

    return train_iter, val_iter
