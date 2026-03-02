import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

num_participants = len(np.unique(participant_ids))
print("Participants detected:", np.unique(participant_ids))
print("X shape:", X_data.shape)
print("y shape:", y_one_hot.shape)

X_data = (X_data - np.mean(X_data, axis=1, keepdims=True)) / \
         (np.std(X_data, axis=1, keepdims=True) + 1e-8)

all_y_true = []
all_y_pred = []

for val_pid in np.unique(participant_ids):

    print(f"\n===== Validating on Participant {val_pid} =====")

    train_idx = np.where(participant_ids != val_pid)[0]
    val_idx = np.where(participant_ids == val_pid)[0]

    X_train, X_val = X_data[train_idx], X_data[val_idx]
    y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]

    print("Train samples:", len(X_train))
    print("Val samples:", len(X_val))

    y_train_int = np.argmax(y_train, axis=1)

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )

    class_weights_dict = dict(zip(np.unique(y_train_int), weights))
    print("Class distribution:", np.bincount(y_train_int))
    print("Class weights:", class_weights_dict)

    model = Sequential([
        Conv1D(64, 5, activation='relu',
               input_shape=(X_data.shape[1], X_data.shape[2])),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, 3, activation='relu'),
        GlobalAveragePooling1D(),

        Dense(64, activation='relu'),
        Dropout(0.5),

        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_probs = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    print("\nFold Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))

    all_y_true.extend(y_true_classes)
    all_y_pred.extend(y_pred_classes)

print("\n===== FINAL LOPO RESULTS =====")
print(classification_report(all_y_true, all_y_pred))
print("Confusion Matrix:")
print(confusion_matrix(all_y_true, all_y_pred))
