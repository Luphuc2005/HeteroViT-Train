"""Pure TensorFlow/Keras implementation of Vision Transformer (ViT) for CIFAR-10."""
from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, Model


class PatchEmbedding(layers.Layer):
    """Extracts patches from images and projects them linearly into embedding dimension."""

    def __init__(self, image_size: int = 32, patch_size: int = 4, embed_dim: int = 192, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = layers.Dense(embed_dim)

        # Learnable CLS token and position embedding
        self.cls_token = None
        self.pos_embedding = None

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, images, training=False):
        batch_size = tf.shape(images)[0]

        # Extract patches using tf.image.extract_patches
        # Output shape: [batch, num_patches_h, num_patches_w, patch_size * patch_size * channels]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Flatten patches: [batch, num_patches, patch_dim]
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dim])

        # Project to embed_dim: [batch, num_patches, embed_dim]
        projected = self.projection(patches)

        # Broadcast CLS token: [batch, 1, embed_dim]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embed_dim])

        # Concatenate CLS token with projected patches: [batch, num_patches + 1, embed_dim]
        tokens = tf.concat([cls_tokens, projected], axis=1)

        # Add positional embedding
        tokens = tokens + self.pos_embedding
        return tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    """Transformer Encoder Block with Pre-LayerNorm architecture."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.dropout1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp_dense1 = layers.Dense(mlp_dim, activation=tf.nn.gelu)
        self.mlp_dropout1 = layers.Dropout(dropout)
        self.mlp_dense2 = layers.Dense(embed_dim)
        self.mlp_dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Attention sub-block (Pre-Norm)
        norm_x = self.norm1(x)
        attn_out = self.mha(norm_x, norm_x, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = x + attn_out

        # MLP sub-block (Pre-Norm)
        norm_x = self.norm2(x)
        mlp_out = self.mlp_dense1(norm_x)
        mlp_out = self.mlp_dropout1(mlp_out, training=training)
        mlp_out = self.mlp_dense2(mlp_out)
        mlp_out = self.mlp_dropout2(mlp_out, training=training)
        x = x + mlp_out

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
        })
        return config


def create_vit_classifier(
    image_size: int = 32,
    patch_size: int = 4,
    num_classes: int = 10,
    embed_dim: int = 192,
    depth: int = 6,
    num_heads: int = 3,
    mlp_dim: int = 768,
    dropout: float = 0.1,
    name: str = "vit_tiny",
) -> Model:
    """Builds a pure TensorFlow/Keras Vision Transformer classifier for CIFAR-10.

    Architecture:
      Image -> Patch extraction & embedding -> CLS token & Positional embedding
            -> Transformer Encoder x depth -> LayerNorm -> CLS representation -> Dense(num_classes)
    """
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image_input")

    # Patch Embedding with CLS token and position embedding
    x = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        name="patch_embedding",
    )(inputs)

    # Transformer Encoder blocks
    for i in range(depth):
        x = TransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"transformer_encoder_{i}",
        )(x)

    # Final LayerNorm
    x = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")(x)

    # Extract CLS token representation (index 0)
    cls_representation = x[:, 0]

    # Classification Head (logits)
    outputs = layers.Dense(num_classes, name="classification_head")(cls_representation)

    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_vit_from_config(config: Dict[str, Any]) -> Model:
    """Builds ViT model from a config dictionary."""
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    return create_vit_classifier(
        image_size=model_cfg.get("image_size", 32),
        patch_size=model_cfg.get("patch_size", 4),
        num_classes=dataset_cfg.get("num_classes", 10),
        embed_dim=model_cfg.get("embed_dim", 192),
        depth=model_cfg.get("depth", 6),
        num_heads=model_cfg.get("num_heads", 3),
        mlp_dim=model_cfg.get("mlp_dim", 768),
        dropout=model_cfg.get("dropout", 0.1),
        name=model_cfg.get("name", "vit_tiny"),
    )
