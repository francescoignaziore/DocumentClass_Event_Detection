import json
import os
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import re
import string
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
sns.set()

print("TF Version: ", tf.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



# Working dirs
# Dataset for Subtask-1
path_to_load_en = '../../03_dataset/task_01/subtask1-document/en-train.json'
path_to_load_es = '../../03_dataset/task_01/subtask1-document/es-train.json'
path_to_load_pr = '../../03_dataset/task_01/subtask1-document/pr-train.json'


def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data

# Load dataset
dataset_df_en = pd.DataFrame(read(path_to_load_en))
dataset_df_es = pd.DataFrame(read(path_to_load_es))
dataset_df_pr = pd.DataFrame(read(path_to_load_pr))

print(dataset_df_en.dtypes)
print(dataset_df_en.head())

print(len(dataset_df_en))
print(len(dataset_df_es))
print(len(dataset_df_pr))

#data.columns
#data['text'][0]
#print(data['text'][0])

# Split text_dataset_from_dir
def prepare_dataset(dataset, batch_size):
    train_x, test_x, train_y, test_y = train_test_split(dataset['text'], dataset['label'], train_size=0.8, shuffle=True, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5, shuffle=True, random_state=42)
    print(len(train_x), 'train examples')
    print(len(val_x), 'validation examples')
    print(len(test_x), 'test examples')
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)
    return train_ds, val_ds, test_ds

train_ds_en, val_ds_en, test_ds_en = prepare_dataset(dataset_df_en, batch_size=32)
train_ds_es, val_ds_es, test_ds_es = prepare_dataset(dataset_df_es, batch_size=32)
train_ds_pr, val_ds_pr, test_ds_pr = prepare_dataset(dataset_df_pr, batch_size=32)

# check train ds
for train_ds in [train_ds_en, train_ds_es, train_ds_pr]:
    print(train_ds)
    for text_batch, label_batch in train_ds.take(1):
        print("Text:\n", text_batch.numpy()[:3])
        print("Event:\n", label_batch.numpy()[:3])


# Standardize text
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)                                              # lowercase
  stripped = tf.strings.regex_replace(lowercase, '\n', '')                              # remove \n
  return tf.strings.regex_replace(stripped,'[%s]' % re.escape(string.punctuation),'')   #  and punct


# Vectorize Text
vocab_size = 1000      # set vocabulary size (10.000)
sequence_length = 200  #  set seq lenght [n x 250]

vectorize_layer = TextVectorization(standardize=custom_standardization,
                                    max_tokens=vocab_size,
                                    output_mode='int',
                                    # ngrams=(1,2)
                                    # output_sequence_length=sequence_length
                                    )
# buid vocabulary
vectorize_layer.adapt(train_ds_en.map(lambda text, label: text))

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

text_batch, label_batch = next(iter(train_ds_en))
# print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
# print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])

# FIXME: issue on win10 https://github.com/tensorflow/tensorflow/issues/43559
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
# print('Vocabulary (first 15): {}'.format(vectorize_layer.get_vocabulary()[:15]))
# workaround
def _get_vocabulary():
    keys, values = vectorize_layer._index_lookup_layer._table_handler.data()
    return [x.decode('utf-8', errors='ignore') for _, x in sorted(zip(values, keys))]

print(_get_vocabulary()[:15])

# vocab = np.array(vectorize_layer.get_vocabulary())
vocab = np.array(_get_vocabulary())
vectorized_example = vectorize_layer(text_batch).numpy()

for n in range(3):
  print("Original:\n", text_batch[n].numpy())
  print("Round-trip:\n", " ".join(vocab[vectorized_example[n]]))
  print()


# Vectorize dataset
train_ds_en = train_ds_en.map(vectorize_text)
val_ds_en = val_ds_en.map(vectorize_text)
test_ds_en = test_ds_en.map(vectorize_text)

test_ds_es = test_ds_es.map(vectorize_text)
test_ds_pr = test_ds_pr.map(vectorize_text)


# Optimize dataflow
train_ds_en = train_ds_en.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds_en = val_ds_en.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds_en = test_ds_en.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


for train_ds in [train_ds_en, train_ds_es, train_ds_pr]:
    print(train_ds)
    for text, label in train_ds.take(1):
        for i in range(3):
            print(f'Embedding:\n {text.numpy()}')
            print(f'Label:\n {label.numpy()}')


# Hyper params
embedding_dim = 32 # 16
epochs = 200

# Define Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001, min_delta=0.001, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=7, verbose=1)

# Select model
# mlp binary classification
# works best without any regularization
def model_MLP(vocab_size, embedding_dim):
    model = tf.keras.Sequential([
      layers.Embedding(input_dim=vocab_size + 1,
                       output_dim=embedding_dim,
                       mask_zero=False,
                       #activity_regularizer=tf.keras.regularizers.L2(l2=0.01)
                       ),
      #layers.Dropout(0.2),
      layers.GlobalAveragePooling1D(),
      #layers.Dropout(0.2),
      layers.Dense(1, activation='sigmoid')])

    return model

model = model_MLP(vocab_size, embedding_dim)
model.summary()

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #'binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(0.01, momentum=0.9, nesterov=True), #tf.keras.optimizers.Adam(0.001), #
              metrics=['accuracy'])

### Train the model
history = model.fit(train_ds_en,
                    validation_data=val_ds_en,
                    epochs=epochs,
                    callbacks=[early_stop, reduce_lr])

### Evaluate the model
loss, accuracy = model.evaluate(test_ds_en)
print('----------------------')
print("    Loss: ", loss)
print("Accuracy: ", accuracy)
print('----------------------')

loss, accuracy = model.evaluate(test_ds_es)
print('----------------------')
print("    Loss: ", loss)
print("Accuracy: ", accuracy)
print('----------------------')

loss, accuracy = model.evaluate(test_ds_pr)
print('----------------------')
print("    Loss: ", loss)
print("Accuracy: ", accuracy)
print('----------------------')



### Create a plot of accuracy and loss over time
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# plot acc, loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# Define decision boundary (map probability to classes)
def decision_boundary(prob):
  return 1 if prob >= .5 else 0

# Count for the metrics
y_hat = model.predict(test_ds_en)
y_hat = [decision_boundary(i) for i in y_hat]
y_true = np.concatenate([i[1].numpy() for i in test_ds_en])

print(confusion_matrix(y_true, y_hat))
print(classification_report(y_true, y_hat))

# Metrics
# Check
true_neg, fals_pos, fals_neg, true_pos = confusion_matrix(y_true, y_hat).ravel()
print(true_pos, fals_pos, true_neg, fals_neg)
accuracy_pos = (true_pos / (true_pos + fals_pos)) * 100
accuracy_neg = (true_neg / (true_neg + fals_neg)) * 100
accuracy_all = (true_pos + true_neg) / (true_pos + true_neg + fals_pos + fals_neg) * 100
print(f'Accuracy of pred positives: {accuracy_pos:.2f} %')
print(f'Accuracy of pred negatives: {accuracy_neg:.2f} %')
print(f'Accuracy: {accuracy_all:.2f} %')



## Export the vectors for Embedding projectpr
def project_model():
    import io

    weights = model.get_layer('embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open('01_vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('02_metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0: continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()



