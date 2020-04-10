""""
Reduce model size: https://github.com/flairNLP/flair/releases

TODO: WTF is THIS???
print(corpus.train[-700013].to_tagged_string('ner'))
print(corpus.train[1].to_tagged_string('pos'))

Songs are in slightly different order from the movie . label <P> machine <P> operator <P> : American International . Publisher : DiJon Music . Liner notes : Joe Bogart & Frank Costa ( WMCA Music Department ) . .
Pedro <P> Galvn <P> is <P> an <P> concreting <P> supervisor <P> . <P> . <P>


"""

from flair.data import Corpus
from flair.datasets import ColumnCorpus
import os

# define columns
columns = {0: 'text', 1: 'pos', 2: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = os.getcwd() + '/dataset'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt',
                              in_memory=False)

len(corpus.train)
print(corpus.train[-700013].to_tagged_string('ner'))
print(corpus.train[1].to_tagged_string('pos'))

# --- #
# https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
# from flair.embeddings import BertEmbeddings, FlairEmbeddings

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('glove'),
    # BertEmbeddings(),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

# trainer: ModelTrainer = ModelTrainer(tagger, corpus)

checkpoint = 'resources/taggers/ner_with_random_dp_1/checkpoint.pt'
trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)

# 7. start training
trainer.train('resources/taggers/ner_with_random_dp_1',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=200,
              checkpoint=True)  # to continue training instead of fresh start


# ---------- #
from flair.data import Sentence

# load the model you trained
model = SequenceTagger.load('resources/taggers/ner_with_random_dp_1/best-model.pt')

# create example sentence
sentence = Sentence('J Elon Reeve Musk FRS is an engineer, industrial designer, and technology entrepreneur. He is a citizen of South Africa, the United States (where he has lived', use_tokenizer=True)

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())

for entity in sentence.get_spans('ner'):
    print(entity, entity.score)

# --- #
from flair.datasets import DataLoader

result, _ = model.evaluate(DataLoader(corpus.test, batch_size=32))
print(result.detailed_results)



# ----------- #
# reduce model size

from flair.inference_utils import WordEmbeddingsStore
from flair.models import SequenceTagger
import pickle

model = SequenceTagger.load('resources/taggers/ner_with_random_dp_1/best-model.pt')
WordEmbeddingsStore.create_stores(model)
pickle.dump(model, open('multi-ner-fast-headless.pickle', 'wb'))

from flair.data import Sentence

model = pickle.load(open('multi-ner-fast-headless.pickle', 'rb'))
WordEmbeddingsStore.load_stores(model)

for _ in range(10):
    sentence = Sentence('Tom is Leasing Manager.', use_tokenizer=True)
    model.predict(sentence)

for entity in sentence.get_spans('ner'):
    print(entity, entity.score)