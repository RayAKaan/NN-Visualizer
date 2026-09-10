"""Pretrained Prediction-Mode Model Registry.

This package provides the centralized, model-agnostic registry + adapter
layer used by Prediction Mode.  Each registered model is represented by a
:class:`ModelAdapter` subclass which is responsible for loading the model,
caching its weights, preprocessing inputs, running inference, post-processing
the output and exposing structured metadata.

The Prediction API should never need model-specific code: it resolves a
``model_id`` against :data:`model_registry.registry.registry`, loads the
adapter lazily and calls the adapter interface.
"""
