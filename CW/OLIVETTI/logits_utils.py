# logits_utils.py
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

def force_build(model: tf.keras.Model, sample_shape: Tuple[int, ...], batch_size: int = 1) -> tf.keras.Model:
    """
    Ensures the model has been 'called' so inbound nodes exist (needed for CleverHans).
    sample_shape should be per-example input shape (e.g., (4096, 1)).
    """
    dummy = np.zeros((batch_size,) + tuple(sample_shape), dtype=np.float32)
    try:
        model.predict(dummy, batch_size=batch_size)
    except Exception:
        try:
            _ = model(dummy, training=False)
        except Exception:
            pass
    return model

def prepare_for_cleverhans(model: tf.keras.Model) -> tf.keras.Model:
    """
    Return a model whose FINAL layer is an explicit Softmax, and whose PREVIOUS layer
    is named 'logits'. CleverHans' KerasModelWrapper will find the softmax layer and
    use the previous layer ('logits') as pre-softmax logits.
    Cases handled:
      A) Final layer is Activation('softmax')      -> ensure previous is named 'logits'
      B) Final Dense has activation=softmax        -> rebuild last Dense (linear) + Softmax
      C) No softmax at end (already logits)        -> append Softmax on top and name prev 'logits'
    """
    last = model.layers[-1]

    # A) Separate Activation('softmax') layer
    if isinstance(last, tf.keras.layers.Activation) and last.activation == tf.keras.activations.softmax:
        # Ensure the previous layer is named 'logits'
        prev = model.layers[-2]
        try:
            prev._name = "logits"
        except Exception:
            # If renaming is not possible due to name conflicts, insert an identity 'logits' layer
            x = prev.output
            logits = layers.Lambda(lambda t: t, name="logits")(x)
            probs = layers.Activation('softmax', name="softmax")(logits)
            return Model(inputs=model.input, outputs=probs, name="probs_model")
        return model

    # B) Inline softmax inside Dense -> split into Dense(linear)+Softmax
    if isinstance(last, tf.keras.layers.Dense) and getattr(last, "activation", None) == tf.keras.activations.softmax:
        inp = model.input
        x = inp
        for lyr in model.layers[:-1]:
            x = lyr(x)
        logits = layers.Dense(last.units, activation=None, name="logits")
        probs  = layers.Activation('softmax', name="softmax")(logits(x))
        new_model = Model(inputs=inp, outputs=probs, name="probs_model")
        # copy weights from original Dense(softmax)
        logits.set_weights(last.get_weights())
        return new_model

    # C) No softmax at end -> append Softmax, and insert an identity layer named 'logits'
    inp = model.input
    x = model.output
    # If the last layer isn't already named 'logits', add an identity layer to guarantee the name
    if getattr(last, "name", None) != "logits":
        logits = layers.Lambda(lambda t: t, name="logits")(x)
    else:
        logits = x
    probs = layers.Activation('softmax', name="softmax")(logits)
    new_model = Model(inputs=inp, outputs=probs, name="probs_model")
    return new_model
