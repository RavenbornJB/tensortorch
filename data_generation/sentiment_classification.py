from keras.datasets import imdb
import random


vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
# print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

# with open("data_imdb_train.txt", mode="w") as f:
#     f.write("25000\n")
#
#     for i in range(25000):
#         X = X_train[i]
#         y = y_train[i]
#
#         f.write(" ".join(map(str, X)))
#         f.write("\n")
#         f.write(str(y))
#         f.write("\n")


# with open("data_imdb_test.txt", mode="w") as f:
#     f.write("2500\n")
#     test_data = list(zip(X_test, y_test))
#     random.shuffle(test_data)
#     X_test_shuffled = [review[0] for review in test_data]
#     y_test_shuffled = [review[1] for review in test_data]
#
#     for i in range(2500):
#         X = X_test_shuffled[i]
#         y = y_test_shuffled[i]
#
#         f.write(" ".join(map(str, X)))
#         f.write("\n")
#         f.write(str(y))
#         f.write("\n")

# print('---review---')
# print(X_train[6])
# print('---label---')
# print(y_train[6])

# word2id = imdb.get_word_index()
#
# id2word = {i: word for word, i in word2id.items()}
#
# with open("vocabulary.txt", mode="w") as f:
#     for id, word in id2word.items():
#         if id >= 5000:
#             continue
#         f.write(f"{id} {word}\n")

# print('---review with words---')
# print([id2word.get(i, ' ') for i in X_train[6]])
# print('---label---')
# print(y_train[6])
