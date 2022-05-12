import tensorflow as tf
from sklearn import linear_model
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras_vectorized


class Model:
    def __init__(self, model_type, x_shape, y_shape):
        """Initialize a model.

        Parameters
        ----------
            model_type : str
                String model name/type.
            x_shape : tuple
                X input datashape for the model.
            y_shape : tuple
                y output datashape for the model.

        Returns
        -------
            None
        """
        self.model_type = model_type
        self.x_shape, self.y_shape = x_shape, y_shape

    def predict(self, x):
        """Return predictions of this model on the given examples.

        Parameters
        ----------
            x : numpy.ndarray
                A numpy array containing inputs to the model.

        Returns
        -------
            y : numpy.ndarray
                A numpy array of predictions.
        """
        raise NotImplementedError

    def fit_sgd(self, x, y, epochs=1):
        """Fit the model with the given data.

        Parameters
        ----------
            x : numpy.ndarray
                A numpy array containing inputs to the model.
            y : numpy.ndarray
                A numpy array of actual predictions corresponding to the given inputs.
            epochs : int
                Number of epochs to train model on (defaults to 1).

        Returns
        -------
            None
        """
        raise NotImplementedError


class LinearRegression(Model):
    def __init__(self, model_type, x_shape, y_shape, lr=0.001):
        Model.__init__(self, model_type, x_shape, y_shape)
        self.sklearn_model = linear_model.LinearRegression()
        self.lr = lr
        self.keras_model = self.build_keras_model()

    def build_keras_model(self):
        model = tf.keras.Sequential(
            [tf.keras.Input(shape=self.x_shape), tf.keras.layers.Dense(self.y_shape)]
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def predict(self, x):
        return self.sklearn_model.predict(x)

    def gradient(self, x, y):
        raise NotImplementedError

    def fit_exact(self, x, y):
        self.sklearn_model.fit(x, y)

    def fit_sgd(self, x, y, epochs=1):
        if hasattr(self.sklearn_model, "coef_"):
            self.keras_model.set_weights(
                [
                    self.sklearn_model.coef_.reshape((-1, self.y_shape)),
                    self.sklearn_model.intercept_.reshape((self.y_shape)),
                ]
            )
        self.keras_model.fit(x, y, epochs=epochs, verbose=0)

        weights = self.keras_model.get_weights()
        if self.y_shape == 1:
            self.sklearn_model.coef_, self.sklearn_model.intercept_ = [
                v.ravel() for v in weights
            ]
        else:
            self.sklearn_model.coef_, self.sklearn_model.intercept_ = weights

    def get_loss(self, x, y):
        losses = np.square(self.sklearn_model.predict(x) - y)
        return losses


class LogisticRegression(Model):
    def __init__(self, model_type, x_shape, y_shape, output_classes, nup, lr=0.0145):
        Model.__init__(self, model_type, x_shape, y_shape)
        self.sklearn_model = linear_model.LogisticRegression(max_iter=10000)
        self.lr = lr
        self.keras_model = self.build_keras_model()
        self.output_classes = output_classes
        self.sklearn_model.classes_ = self.output_classes
        self.nup = nup

    def build_keras_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=self.x_shape),
                tf.keras.layers.Dense(self.y_shape),
            ]
        )

        opt = tf.keras.optimizers.SGD(learning_rate=self.lr)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=opt,
            metrics=["accuracy"],
        )
        return model

    def predict(self, x):
        return self.sklearn_model.predict(x)

    def gradient(self, x, y):
        raise NotImplementedError

    def fit_sgd(
        self,
        x,
        y,
        epochs=1,
        retrain=None,
        sgd_only=False,
    ):
        if sgd_only:
            tf.keras.backend.set_value(
                self.keras_model.optimizer.learning_rate, self.lr / 100
            )

        if hasattr(self.sklearn_model, "coef_"):
            self.keras_model.set_weights(
                [
                    self.sklearn_model.coef_.T,
                    self.sklearn_model.intercept_.reshape((self.y_shape)),
                ]
            )

        self.keras_model.fit(
            x,
            y,
            batch_size=self.nup,
            epochs=epochs,
            verbose=0,
        )

        weights = self.keras_model.get_weights()
        self.sklearn_model.coef_, self.sklearn_model.intercept_ = np.array(
            weights[0].T
        ), np.array(weights[1])

    def get_loss(self, x, y):
        probs = self.sklearn_model.predict_proba(x)
        nlog_probs = -np.log(probs)
        xe_loss = (nlog_probs * np.eye(self.y_shape)[y]).sum(axis=1)
        return xe_loss

    def score(self, x, y):
        return self.sklearn_model.score(x, y)


class ImageNNet(Model):
    def __init__(
        self,
        model_type,
        x_shape,
        y_shape,
        output_classes,
        nup,
        lr=0.0001,
    ):
        Model.__init__(self, model_type, x_shape, y_shape)
        self.nup = nup
        self.output_classes = output_classes
        self.lr = lr
        self.keras_model = self.build_keras_model(model_type)

    def build_keras_model(self, model_type):
        if model_type == "2f":
            model = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=self.x_shape),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(self.y_shape, activation="softmax"),
                ]
            )
            # want 0.01 for lr
            opt = tf.keras.optimizers.SGD(learning_rate=self.lr)
        elif model_type == "rnft":
            pre_model = tf.keras.applications.VGG16(
                include_top=False, input_shape=self.x_shape, pooling="avg"
            )
            model = tf.keras.Sequential(
                [
                    pre_model,
                    tf.keras.layers.Dense(
                        self.y_shape,
                        kernel_regularizer=tf.keras.regularizers.l2(l=0),
                        activation="softmax",
                    ),
                ]
            )
            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        else:
            raise NotImplementedError

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=opt,
            metrics=["accuracy"],
        )
        return model

    def predict(self, x):
        return self.keras_model.predict_proba(x)

    def fit_sgd(
        self,
        x,
        y,
        test=None,
        epochs=1,
        retrain=None,
        sgd_only=False,
    ):
        if sgd_only:
            tf.keras.backend.set_value(
                self.keras_model.optimizer.learning_rate, self.lr / 10
            )

        self.keras_model.fit(
            x,
            y,
            batch_size=self.nup,
            epochs=epochs,
            verbose=0,
            validation_data=test,
        )

    def get_loss(self, x, y):
        return self.keras_model.evaluate(x, y, verbose=0)[0]

    def score(self, x, y):
        return self.keras_model.evaluate(x, y, verbose=0)[1]


class DPLogisticRegression(LogisticRegression):
    def __init__(
        self,
        model_type,
        x_shape,
        y_shape,
        output_classes,
        nup,
        noise_multiplier=None,
        l2_norm_clip=None,
        lr=0.0145,
    ):
        super().__init__(
            model_type=model_type,
            x_shape=x_shape,
            y_shape=y_shape,
            output_classes=output_classes,
            nup=nup,
            lr=lr,
        )
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip

    def fit_sgd(
        self,
        x,
        y,
        test=None,
        epochs=1,
        retrain=False,
        sgd_only=False,
    ):
        if retrain:
            opt = dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=self.lr * 10,
                learning_rate=self.lr,
            )

            self.keras_model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True, reduction=tf.losses.Reduction.NONE
                ),
                optimizer=opt,
                metrics=["accuracy"],
            )

        super().fit_sgd(
            x=x,
            y=y,
            epochs=epochs,
            sgd_only=sgd_only,
        )


class BERTSentiment(Model):
    def __init__(
        self,
        model_type,
        x_shape,
        y_shape,
        nup,
        tokenizer,
        data_collator,
        output_dir="imdb/saved_models",
        lr=1e-5,
    ):
        Model.__init__(self, model_type, x_shape, y_shape)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        self.output_dir = output_dir
        self.lr = lr
        self.nup = nup
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    def build_keras_model(self):
        return NotImplementedError

    def predict(self, dl):
        self.model.to(0)
        self.model.eval()

        predictor = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        return predictor.predict(dl).predictions

    def fit_sgd(self, train_dl, epochs=1, retrain=False, sgd_only=False, test=None):
        self.model.to(0)

        if sgd_only:
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                learning_rate=self.lr / 10,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=epochs,
                weight_decay=0.01,
            )
        else:
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                learning_rate=self.lr,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=epochs,
                weight_decay=0.01,
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dl,
            eval_dataset=test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()

    def get_loss(self, dl):
        self.model.to(0)
        self.model.eval()

        loss = torch.nn.CrossEntropyLoss(reduction="none", reduce=False)
        softmax = torch.nn.Softmax(dim=1)

        prediction = softmax(torch.tensor(self.predict(dl)))
        labels = dl["label"]

        return loss(prediction, torch.tensor(labels)).numpy()

    def score(self, dl):
        self.model.to(0)
        self.model.eval()

        prediction = self.predict(dl)
        preds = torch.argmax(torch.tensor(prediction), dim=1)
        labels = dl["label"]

        return accuracy_score(preds.numpy(), labels)
