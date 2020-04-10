"""
TODO:
1. Expand the dataset by randomizing professions by adding new datapoints !!!
2. Add noise to resemble google search items more


TODO: generally the dataset is not really good, lots of false pasitives and unlabeled data...
1. Try out another dataset WikiData
"""

import random
import re

# finding consecutive ints in a list
from itertools import groupby
from operator import itemgetter

# refactor and read
dataset = []
for file in ['EWNERTC_TC_Fine Grained NER_No_NoiseReduction.DUMP', 'EWNERTC_TC_Fine Grained NER_DomainDependent_NoiseReduction.DUMP']:
    with open(file, 'rb') as f:
        for line in f:
            line = line.decode("utf-8")
            line = line.replace('\r', '').replace('\n', '')
            domain, tags, sentence = line.split('\t')
            if 'profession' in tags:
                dataset.append([domain, tags, sentence])
                # print(domain, sentence)

# read and clean addition professions
other_titles = open('dataset/titles_combined.txt').read().splitlines()

max_parts = 0
for idx, title in enumerate(other_titles):
    if ',' in title:
        title_parts = title.split(', ')
        other_titles[idx] = ' '.join(reversed(title_parts)).lower()
        max_parts = [max_parts if max_parts > len(title_parts) else len(title_parts)][0]
        print(other_titles[idx])

for idx, title in enumerate(other_titles):
    other_titles[idx] = title.lower()


# refactor and write
splits = [0, int(len(dataset) * 0.70)], \
         [int(len(dataset) * 0.70), int(len(dataset) * 0.85)], \
         [int(len(dataset) * 0.85), int(len(dataset) * 1)]

random.shuffle(dataset)
for dataset_name, split in zip(['train', 'test', 'dev'], splits):
    open('dataset/{}.txt'.format(dataset_name), 'w').close()  # remove everything
    with open('dataset/{}.txt'.format(dataset_name), 'a') as f:
        for sentence in dataset[split[0]:split[1]]:
            _, tags, words = sentence

            # -------------------------------------------- #
            # random modification by concatinating more professions in a sentence
            add_random = random.randint(0, 100) < 30
            if add_random and dataset_name == 'train':

                words_list = words.split(' ')
                tags_list = tags.split(' ')

                # pick last one because else can insert in the middle of two word profession
                target_index = [idx for idx, s in enumerate(tags_list) if 'profession' in s][-1] + 1
                initial_target_index = target_index
                random_titles = random.sample(other_titles, 1)

                for random_title in random_titles:

                    new_words_list = words_list[:target_index] + [random.choice([',', 'and'])]
                    new_tags_list = tags_list[:target_index] + ['O']
                    target_index += 1

                    random_title = random_title.replace('-', ' ').lower()
                    for random_title_part in random_title.split(' '):
                        new_words_list = new_words_list[:target_index] + [random_title_part]
                        new_tags_list = new_tags_list[:target_index] + ['randomAdd-profession']
                        target_index += 1

                new_words_list = new_words_list + words_list[initial_target_index:]
                new_tags_list = new_tags_list + tags_list[initial_target_index:]

                words = ' '.join(new_words_list)
                tags = ' '.join(new_tags_list)

            for tag, word in zip(tags.split(' '), words.split(' ')):

                # https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md
                if 'profession' in tag:
                    line = word + '\t' + 'P' + '\t' + 'P'
                else:
                    line = word + '\t' + 'P' + '\t' + 'O'
                f.write(line + '\n')
            f.write(line + '\n\n')

            # -------------------------------------------- #
            # replace all existing professions with new ones
            words_list = words.split(' ')
            tags_list = tags.split(' ')
            targets_indices = [idx for idx, s in enumerate(tags_list) if 'profession' in s]

            # find consecutive tags of professions
            profession_groups = []
            for k, g in groupby(enumerate(targets_indices), lambda ix: ix[0] - ix[1]):
                profession_groups.append(list(map(itemgetter(1), g)))

            # strip the sentence of the professions (wil add in the next step)
            # this way it is simpler, due to different lengths of professions
            # e.g. might strip off profession of 3 words and add profession only one word
            profession_idx_flat = [item for sublist in profession_groups for item in sublist]
            new_words_list = [i for j, i in enumerate(words_list) if j not in profession_idx_flat]
            new_tags_list = [i for j, i in enumerate(tags_list) if j not in profession_idx_flat]

            # these are the indeces of where to insert new professions
            placeholder_idx = [i[0] for i in profession_groups]
            placeholder_idx = [i + 1 - len(j) for i, j in zip(placeholder_idx, profession_groups)]

            idx_add = 0
            for idx in placeholder_idx:

                idx += idx_add
                random_title = random.sample(other_titles, 1)[0].split(' ')
                n_words = len(random_title)

                new_words_list = new_words_list[:idx] + random_title + new_words_list[idx:]
                new_tags_list = new_tags_list[:idx] + ['expandRandom-profession'] * n_words + new_tags_list[idx:]

                idx_add += n_words - 1

                words = ' '.join(new_words_list)
                tags = ' '.join(new_tags_list)

            # check if correct
            # print(len(new_words_list), len(new_tags_list))
            # print(' '.join(new_words_list))
            # print([word for word, tag in zip(new_words_list, new_tags_list) if 'profession' in tag])

            for tag, word in zip(tags.split(' '), words.split(' ')):

                # https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md
                if 'profession' in tag:
                    line = word + '\t' + 'P' + '\t' + 'P'
                else:
                    line = word + '\t' + 'P' + '\t' + 'O'
                f.write(line + '\n')
            f.write(line + '\n\n')


# TODO: I-person_profession vs B-person_profession vs B-profession_specialization_of etc
unique_set = set()
for item in dataset:
    if 'profession' in item[1]:
        print(item)

for tag in unique_set:
    if 'profession' in tag:
        print(tag)


# TODO: check inputs one by one
with open('dataset/train.txt', 'r') as f:
    for line in f:
        if 'analyst' in line.lower():
            print(line)