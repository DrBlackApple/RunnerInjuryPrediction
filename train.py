import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    roc_auc_score,
)
import numpy as np
import seaborn as sns

from libnn.model import Model, ModelCheckpoint
from libnn.layers import Dense, Dropout, Input
from libnn.functions import relu, glorot_uniform, binary_crossentropy_loss, sigmoid
from libnn.sgd import SGD

from load_data import get_data

SAVED_MODEL = "model_best.dat"

IMG_FOLDER = "img/"
SAVED_LOSS_CURB = f"{IMG_FOLDER}losses.png"
SAVED_CONFUSION_MATRIX = f"{IMG_FOLDER}confusion_matrix.png"
SAVEC_ROC_CURB = f"{IMG_FOLDER}roc.png"


def evaluate(X_test: np.ndarray, y_test: np.ndarray):
    with open(SAVED_MODEL, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test).reshape((-1, 1))
    print(y_pred.shape, y_test.shape)

    auc_score = roc_auc_score(y_test, y_pred)
    print("AUC :", auc_score)
    print("Accuracy :", accuracy_score(y_test, y_pred.round()))

    fpr, tpr, _ = roc_curve(y_test, y_pred)

    # threshold activation at 50%
    y_pred = np.where(y_pred > 0.5, 1, 0)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.savefig(SAVED_CONFUSION_MATRIX)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.show()
    plt.savefig(SAVEC_ROC_CURB)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()

    model = Model(
        [
            Input(X_train.shape[-1]),
            Dense(256, activation=relu, initializer=glorot_uniform),
            Dropout(0.2),
            Dense(256, activation=relu, initializer=glorot_uniform),
            Dropout(0.2),
            Dense(256, activation=relu, initializer=glorot_uniform),
            Dropout(0.2),
            Dense(64, activation=relu, initializer=glorot_uniform),
            Dropout(0.15),
            Dense(1, activation=sigmoid),
        ]
    )

    model.build(
        loss_function=binary_crossentropy_loss,
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
    )

    model.summary()

    callback = [ModelCheckpoint(filename=SAVED_MODEL)]

    loss = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        callbacks=callback,
    )

    plt.figure()
    plt.plot(loss, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(SAVED_LOSS_CURB)

    evaluate(X_test, y_test)
