import tensorflow as tf
from typing import Tuple

@tf.keras.utils.register_keras_serializable(package="market_regime_discovery", name="MaskedReconstruction")
class MaskedReconstruction(tf.keras.layers.Layer):
    def __init__(self, mask_prob=0.15, **kwargs):
        super().__init__(**kwargs)
        self.mask_prob = mask_prob
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        mask = tf.random.uniform(tf.shape(inputs)[:-1]) > self.mask_prob
        mask = tf.expand_dims(mask, axis=-1)
        return tf.where(mask, inputs, 0.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({"mask_prob": self.mask_prob})
        return config

@tf.keras.utils.register_keras_serializable(package="market_regime_discovery", name="FeatureSplitter")
class FeatureSplitter(tf.keras.layers.Layer):
    def __init__(self, shape_indices, energy_indices, **kwargs):
        super().__init__(**kwargs)
        self.shape_indices = shape_indices
        self.energy_indices = energy_indices
    
    def call(self, inputs):
        shape_input = tf.gather(inputs, self.shape_indices, axis=-1)
        energy_input = tf.gather(inputs, self.energy_indices, axis=-1)
        return shape_input, energy_input
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "shape_indices": self.shape_indices,
            "energy_indices": self.energy_indices
        })
        return config

def build_decoupled_autoencoder(
    input_shape: Tuple[int, int],
    latent_dim: int,
    feature_groups: dict,
    temporal_lambda: float = 0.0,
    multi_horizon_lambda: float = 0.0,
    contrastive_weight: float = 0.0
):
    window_size, n_features = input_shape
    inputs = tf.keras.Input(shape=input_shape)
    
    masked_inputs = MaskedReconstruction()(inputs)
    
    all_cols = feature_groups["all_cols"]
    shape_idx = [i for i, col in enumerate(all_cols) if col in feature_groups["shape_cols"]]
    energy_idx = [i for i, col in enumerate(all_cols) if col in feature_groups["energy_cols"]]
    shape_input, energy_input = FeatureSplitter(shape_idx, energy_idx)(masked_inputs)
    
    x_shape = tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(shape_input)
    x_shape = tf.keras.layers.Conv1D(8, kernel_size=3, activation='relu', padding='same')(x_shape)
    x_shape = tf.keras.layers.GlobalAveragePooling1D()(x_shape)
    
    x_energy = tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(energy_input)
    x_energy = tf.keras.layers.Conv1D(8, kernel_size=3, activation='relu', padding='same')(x_energy)
    x_energy = tf.keras.layers.GlobalAveragePooling1D()(x_energy)
    
    merged = tf.keras.layers.Concatenate()([x_shape, x_energy])
    encoded_raw = tf.keras.layers.Dense(latent_dim, name='bottleneck')(merged)
    
    def l2_normalize_with_floor(v):
        v_norm = tf.nn.l2_normalize(v, axis=1)
        return v_norm + 1e-4
    
    encoded = tf.keras.layers.Lambda(
        l2_normalize_with_floor,
        output_shape=(latent_dim,)
    )(encoded_raw)

    half_len = window_size // 2
    x = tf.keras.layers.Dense(half_len * 16, activation='relu')(encoded)
    x = tf.keras.layers.Reshape((half_len, 16))(x)
    x = tf.keras.layers.Conv1DTranspose(16, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    decoded = tf.keras.layers.Conv1D(n_features, kernel_size=3, padding='same')(x)
    
    if decoded.shape[1] != window_size:
        decoded = tf.keras.layers.Lambda(lambda t: t[:, :window_size, :])(decoded)

    autoencoder = tf.keras.Model(inputs, decoded, name="DecoupledAutoencoder")
    encoder = tf.keras.Model(inputs, encoded, name="Encoder")
    autoencoder.encoder = encoder
    autoencoder.temporal_lambda = temporal_lambda
    autoencoder.multi_horizon_lambda = multi_horizon_lambda
    autoencoder.contrastive_weight = contrastive_weight
    return autoencoder

def temporal_loss(autoencoder, x_clean, x_recon, z):
    recon_loss = tf.reduce_mean(tf.keras.losses.Huber()(x_clean, x_recon))
    total_loss = recon_loss
    
    if autoencoder.temporal_lambda > 0 and tf.shape(z)[0] >= 2:
        z_diff = z[1:] - z[:-1]
        temporal_loss = autoencoder.temporal_lambda * tf.reduce_mean(tf.square(z_diff))
        total_loss += temporal_loss
    
    if autoencoder.multi_horizon_lambda > 0 and tf.shape(z)[0] >= 3:
        z_diff_mh = z[2:] - z[:-2]
        mh_loss = autoencoder.multi_horizon_lambda * tf.reduce_mean(tf.square(z_diff_mh))
        total_loss += mh_loss
    
    if autoencoder.contrastive_weight > 0 and tf.shape(z)[0] >= 3:
        pos_dist = tf.reduce_sum(tf.square(z[1:] - z[:-1]), axis=1)
        neg_dist = tf.reduce_sum(tf.square(z[2:] - z[:-2]), axis=1)
        pos_dist_aligned = pos_dist[:-1]
        contrastive_loss = autoencoder.contrastive_weight * tf.reduce_mean(
            tf.maximum(pos_dist_aligned - neg_dist + 1.0, 0.0)
        )
        total_loss += contrastive_loss
    
    return total_loss