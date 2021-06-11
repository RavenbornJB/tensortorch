from keras.datasets import imdb
import random


vocabulary_size = 800
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
# print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

train_examples = 2000

with open("data_imdb_train.txt", mode="w") as f:
    f.write(f"{train_examples}\n")

    train_data = list(zip(X_train, y_train))
    random.shuffle(train_data)
    train_data = train_data[:train_examples]
    train_data.sort(key=lambda x: len(x[0]))

    X_train_shuffled = [item[0] for item in train_data]
    y_train_shuffled = [item[1] for item in train_data]

    for i in range(train_examples):
        X = X_train_shuffled[i]
        y = y_train_shuffled[i]

        f.write(" ".join(map(str, X)))
        f.write("\n")
        f.write(str(y))
        f.write("\n")


test_examples = 200

with open("data_imdb_test.txt", mode="w") as f:
    f.write(f"{test_examples}\n")
    test_data = list(zip(X_test, y_test))
    random.shuffle(test_data)
    test_data = test_data[:test_examples]
    test_data.sort(key=lambda x: len(x[0]))

    X_test_shuffled = [item[0] for item in test_data]
    y_test_shuffled = [item[1] for item in test_data]

    for i in range(test_examples):
        X = X_test_shuffled[i]
        y = y_test_shuffled[i]

        f.write(" ".join(map(str, X)))
        f.write("\n")
        f.write(str(y))
        f.write("\n")


word2id = imdb.get_word_index()

id2word = {i: word for word, i in word2id.items()}

with open("vocabulary.txt", mode="w") as f:
    for id, word in id2word.items():
        if id >= vocabulary_size:
            continue
        f.write(f"{id} {word}\n")
