from gensim.models import KeyedVectors


# Import word2vec model.
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)

# Some sentences to test.
# sentence_one = 'Obama speaks to media in Illinois'.lower().split()
# sentence_two = 'The president greets press in Chicago'.lower().split()

sentence_one = input("First sentence is... ")
sentence_two = input("Second sentence is... ")

# Compute WMD.
distance = model.wmdistance(sentence_one, sentence_two)
print(distance)