# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Utilities for sampling room images"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from spark_config import Config, config_field, register_config

from hvlm_planner_python.misc import LoggerWarn, default_device

import semantic_inference_python.models


@dataclass
class ImageSamplingConfig(Config):
    """Configuration for image sampling."""

    max_images_per_room: int = 5
    clip_model: Any = config_field("clip", default="open_clip")


class BaseImageSampler:
    """Base class for image samplers."""

    def __init__(self, config) -> None:
        """Base constructor for image samplers.
        :param config: Configuration for the image sampler.
        """
        self._config = config
        self._encoder = self._config.clip_model.create()
        self._encoder.to(default_device(cuda_device=0))

        self._object_features = None
        self._question_feature = None

    def set_objects(self, objects: np.ndarray) -> None:
        """Set object features for the image sampler.
        :param objects: List of object features.
        """
        self._object_features = objects.astype(np.float32)
        if self._object_features.ndim == 1:
            self._object_features = self._object_features[None, :]

        obj_norms = np.linalg.norm(self._object_features, axis=1, keepdims=True)
        obj_norms[obj_norms == 0] = 1.0
        self._object_features /= obj_norms

    def set_question(self, question: str) -> None:
        """Set question feature for the image sampler.
        :param question: Question string.
        """
        self._question_feature = (
            self._encoder.embed_text([question]).cpu().numpy().squeeze()
        )
        self._question_feature = self._question_feature.astype(np.float32)
        if self._question_feature.ndim == 1:
            self._question_feature = self._question_feature[None, :]

        q_norm = np.linalg.norm(self._question_feature, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        self._question_feature /= q_norm

    def _checks(
        self, images: list[np.ndarray], image_features: list[np.ndarray]
    ) -> tuple[bool, list[np.ndarray]]:
        """Perform checks on the input images and features.
        :param images: List of images for the room.
        :param image_features: List of feature vectors corresponding to the images.
        :return: Tuple of (is_valid, processed_images)
        """
        if not images:
            LoggerWarn.warning("No images provided for room, returning empty list")
            return False, []

        if not image_features:
            LoggerWarn.warning("No image features provided, returning all images")
            return False, images[: self._config.max_images_per_room]

        if len(images) != len(image_features):
            LoggerWarn.warning(
                "Number of images and image features do not match, returning all images"
            )
            return False, images[: self._config.max_images_per_room]

        return True, []

    def sample_images(
        self, images: list[np.ndarray], image_features: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Sample a subset of images based on the configuration.
        :param images: List of images for the room.
        :param image_features: List of feature vectors corresponding to the images.
        :return: List of sampled images.
        """
        raise NotImplementedError("sample_images must be implemented by subclasses.")


class DisparityImageSampler(BaseImageSampler):
    """Image sampler that prioritizes images with high feature disparity."""

    def __init__(self, config) -> None:
        """Constructor for DisparityImageSampler.
        :param config: Configuration for the image sampler.
        """
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "DisparityImageSampler":
        """Load model from configuration dictionary."""
        config = DisparityImageSamplerConfig()
        config.update(kwargs)
        return cls(config)

    def sample_images(
        self, images: list[np.ndarray], image_features: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Sample a subset of images based on feature disparity.
        :param images: List of images for the room.
        :param image_features: List of feature vectors corresponding to the images.
        :return: List of sampled images.
        """
        is_valid, fallback_images = self._checks(images, image_features)
        if not is_valid:
            return fallback_images
        n = len(images)
        max_images = min(self._config.max_images_per_room, n)

        features = np.vstack(image_features).astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features /= norms

        similarity = features @ features.T
        distance = 1.0 - similarity

        selected_mask = np.zeros(n, dtype=bool)

        # Random first pick
        first_idx = np.random.randint(n)
        selected_mask[first_idx] = True
        min_distances = distance[first_idx].copy()

        for _ in range(1, max_images):
            min_distances[selected_mask] = -1.0

            next_idx = np.argmax(min_distances)
            selected_mask[next_idx] = True

            min_distances = np.minimum(min_distances, distance[next_idx])

        selected_indices = np.where(selected_mask)[0]
        return [images[i] for i in selected_indices]


@register_config("image_sampling", name="disparity", constructor=DisparityImageSampler)
@dataclass
class DisparityImageSamplerConfig(ImageSamplingConfig):
    """Configuration for DisparityImageSampler."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class ObjectImageSampler(BaseImageSampler):
    """Image sampler that prioritizes images based on object presence."""

    def __init__(self, config) -> None:
        """Constructor for ObjectImageSampler.
        :param config: Configuration for the image sampler.
        """
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "ObjectImageSampler":
        """Load model from configuration dictionary."""
        config = ObjectImageSamplerConfig()
        config.update(kwargs)
        return cls(config)

    def sample_images(
        self, images: list[np.ndarray], image_features: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Sample a subset of images based on object presence.
        :param images: List of images for the room.
        :param image_features: List of feature vectors corresponding to the images.
        :return: List of sampled images.
        """
        is_valid, fallback_images = self._checks(images, image_features)
        if not is_valid:
            return fallback_images

        if self._object_features is None:
            LoggerWarn.warning("Object features not set, returning first images")
            return images[: self._config.max_images_per_room]

        n = len(images)
        max_images = min(self._config.max_images_per_room, n)

        img_feats = np.vstack(image_features).astype(np.float32)

        # Normalize imag features
        img_norms = np.linalg.norm(img_feats, axis=1, keepdims=True)
        img_norms[img_norms == 0] = 1.0
        img_feats /= img_norms

        # Cosine similarity: (n_images, n_objects)
        similarity = img_feats @ self._object_features.T

        # Take max over objects
        max_similarity = np.max(similarity, axis=1)

        # Select top-K indices
        top_indices = np.argsort(-max_similarity)[:max_images]

        return [images[i] for i in top_indices]


@register_config("image_sampling", name="object", constructor=ObjectImageSampler)
@dataclass
class ObjectImageSamplerConfig(ImageSamplingConfig):
    """Configuration for ObjectImageSampler."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class QuestionImageSampler(BaseImageSampler):
    """Image sampler that prioritizes images based on question relevance."""

    def __init__(self, config) -> None:
        """Constructor for QuestionImageSampler.
        :param config: Configuration for the image sampler.
        """
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "QuestionImageSampler":
        """Load model from configuration dictionary."""
        config = QuestionImageSamplerConfig()
        config.update(kwargs)
        return cls(config)

    def sample_images(
        self, images: list[np.ndarray], image_features: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Sample a subset of images based on question relevance.
        :param images: List of images for the room.
        :param image_features: List of feature vectors corresponding to the images.
        :return: List of sampled images.
        """
        is_valid, fallback_images = self._checks(images, image_features)
        if not is_valid:
            return fallback_images

        if self._question_feature is None:
            LoggerWarn.warning("Question feature not set, returning first images")
            return images[: self._config.max_images_per_room]

        n = len(images)
        max_images = min(self._config.max_images_per_room, n)

        # Stack image features
        img_feats = np.vstack(image_features).astype(np.float32)

        # Normalize image features
        img_norms = np.linalg.norm(img_feats, axis=1, keepdims=True)
        img_norms[img_norms == 0] = 1.0
        img_feats /= img_norms

        # Cosine similarity (n_images,)
        similarity = (img_feats @ self._question_feature.T).squeeze()

        # Select top-K indices
        top_indices = np.argsort(-similarity)[:max_images]

        return [images[i] for i in top_indices]


@register_config("image_sampling", name="question", constructor=QuestionImageSampler)
@dataclass
class QuestionImageSamplerConfig(ImageSamplingConfig):
    """Configuration for QuestionImageSampler."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
