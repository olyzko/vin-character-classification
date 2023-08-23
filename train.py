import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras import layers
from keras.utils import np_utils
from extra_keras_datasets import emnist
from sklearn.model_selection import train_test_split


def delete_extra_letters(X, y):

    # Indices of letters I,O,Q and lowercase letters, which are not used in VIN
    extra_values_set = {18, 24, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46}
    indices = np.where(np.isin(y, list(extra_values_set)))
    indices = indices[0]
    X = np.delete(X, indices, axis=0)
    y = np.delete(y, indices)

    X = X.astype('float32') / 255
    X = X.reshape(-1, 28, 28, 1)

    # Reindexing target variable
    sorted_unique_values = np.unique(y)
    mapping_dict = {value: index for index, value in enumerate(sorted_unique_values)}
    y = np.array([mapping_dict[value] for value in y])
    return X, y


(X_train, y_train), (X_test, y_test) = emnist.load_data(type='balanced')

X_train, y_train = delete_extra_letters(X_train, y_train)
X_test, y_test = delete_extra_letters(X_test, y_test)
print(X_train.shape)

y_train = np_utils.to_categorical(y_train, 33)
y_test = np_utils.to_categorical(y_test, 33)

X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.15,
                                                  random_state=35)


model = Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='tanh'))
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(33, activation='softmax'))

optimizer_name = 'adam'
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')
mcp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', verbose=1, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.15, callbacks=[early_stopping, mcp_save])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy:}')

model.save("model.h5")
