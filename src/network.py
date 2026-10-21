"""One frozen backbone and the two established residual branches."""
from tensorflow import keras
from tensorflow.keras import layers
from common import BOTTLENECK


class Backbone:
    def __init__(self, path):
        self.original = keras.models.load_model(path, compile=False)
        self.sequence = keras.Model(self.original.inputs[0], self.original.get_layer('Dense1').output,
                                    name='CRISPRonABE_sequence_encoder')
        self.sequence.trainable = False

    def build(self, fold_seed, length_weights=None):
        structural = length_weights is not None
        inputs = [keras.Input(shape, name=name) for shape, name in [
            ((30, 4), 'one_hot'), ((8,), 'outcome_properties'), ((1,), 'energy_properties'),
            ((1,), 'cas9'), ((5,), 'dataset'), ((1,), 'length_properties')]]
        fusion = layers.Concatenate(name='OriginalFusion')(
            [self.sequence(inputs[0], training=False), *inputs[1:5]])
        old = self.original.get_layer('Dense2')
        frozen = layers.Dense(old.units, trainable=False, name='FrozenDense2Linear')
        base = frozen(fusion)
        frozen.set_weights(old.get_weights())
        length = layers.Dense(old.units, kernel_initializer='glorot_uniform' if structural else 'zeros', bias_initializer='zeros',
                              trainable=not structural,
                              name='FrozenLengthToDense2' if structural else 'LengthToDense2')
        correction = length(inputs[5])
        if structural:
            length.set_weights(length_weights)
        corrections = [base, correction]
        if structural:
            distance = keras.Input((8,), name='delta_distance_properties')
            inputs.append(distance)
            latent = layers.Dense(BOTTLENECK, activation='tanh', use_bias=False,
                kernel_initializer=keras.initializers.GlorotUniform(seed=fold_seed + 1),
                name='DeltaDistanceBottleneck')(distance)
            delta = layers.Dense(old.units, use_bias=False, kernel_initializer='zeros',
                                 name='DeltaDistanceToDense2')(latent)
            # Preserve the established graph: the ungated scalar remains exactly one.
            gate_raw = layers.Dense(1, activation='tanh', kernel_initializer='zeros',
                                    bias_initializer='zeros', trainable=False, name='LengthGateRaw')(inputs[5])
            gate = layers.Lambda(lambda x: 1.0 + x, name='LengthGate')(gate_raw)
            corrections.append(layers.Multiply(name='GatedDeltaDistance')([delta, gate]))
        hidden = layers.Add(name='Dense2Preactivation')(corrections)
        hidden = layers.Activation(old.activation, name='Dense2Activation')(hidden)
        for source, name in [('Dense3', 'FrozenDense3'), ('Output', 'FrozenOutput')]:
            old = self.original.get_layer(source)
            layer = layers.Dense(old.units, activation=old.activation, trainable=False, name=name)
            hidden = layer(hidden)
            layer.set_weights(old.get_weights())
        return keras.Model(inputs, hidden)
