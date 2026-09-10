"""Registry discovery, metadata & status tests (no network / no inference)."""
import pytest

from model_registry.base import ModelLoadError
from model_registry.registry import UNAVAILABLE_MODELS, registry
from model_registry.schema import (
    FAMILY_ANN,
    FAMILY_CNN,
    FAMILY_RNN,
    MODEL_META_FIELDS,
)


def _meta_of(model_id):
    for row in registry.catalog():
        if row.get("id") == model_id:
            return row
    raise AssertionError(f"{model_id} not in catalog")


def test_registry_discovers_all_families():
    ids = registry.ids()
    assert "ann-dacorvo-mnist-mlp" in ids
    assert "ann-mysticdan-mlp-mnist" in ids
    assert "ann-nn-visualizer" in ids
    assert "cnn-mobilenetv3small" in ids
    assert "cnn-mobilenetv2" in ids
    assert "cnn-resnet50" in ids
    assert "cnn-vgg16" in ids
    assert "rnn-bilstm-imdb-kerasio" in ids
    assert registry.families() == [FAMILY_ANN, FAMILY_CNN, FAMILY_RNN]


def test_registry_metadata_schema():
    rows = [r for r in registry.catalog() if r["status"] != "unavailable"]
    assert len(rows) >= 7
    for row in rows:
        for field in MODEL_META_FIELDS:
            assert field in row, f"{row.get('id')} missing '{field}'"
        assert row["pretrained"] is True
        assert row["family"] in (FAMILY_ANN, FAMILY_CNN, FAMILY_RNN)


def test_ann_model_details():
    m = _meta_of("ann-dacorvo-mnist-mlp")
    assert m["input_type"] == "mnist_pixels"
    assert m["num_classes"] == 10
    assert m["parameter_count"] == 269_322
    assert "hugging" in m["source"].lower()
    assert "huggingface.co" in m["weights"].lower()

    mm = _meta_of("ann-mysticdan-mlp-mnist")
    assert mm["architecture"].startswith("MLP 784-512-256-128-64-10")
    assert mm["parameter_count"] == 575_050


def test_cnn_model_details():
    for cid, expected in {
        "cnn-mobilenetv3small": 2_554_968,
        "cnn-mobilenetv2": 3_538_984,
        "cnn-resnet50": 25_636_712,
        "cnn-vgg16": 138_357_544,
    }.items():
        m = _meta_of(cid)
        assert m["input_type"] == "image"
        assert m["num_classes"] == 1000
        assert m["parameter_count"] == expected
        assert m["weights"] == "imagenet (Keras cache, auto-downloaded on first load)"


def test_rnn_model_details():
    m = _meta_of("rnn-bilstm-imdb-kerasio")
    assert m["input_type"] == "text"
    assert m["num_classes"] == 2
    assert m["parameter_count"] == 207_585
    assert "IMDB" in m["dataset"]


def test_incompatible_user_models_are_listed_unavailable():
    ids = {u["id"] for u in UNAVAILABLE_MODELS}
    assert "rnn-jongador-lstm-imdb-256" in ids
    assert "rnn-jongador-lstm-imdb-512" in ids
    assert "rnn-pratyushee-assamese-lstm" in ids
    for row in registry.catalog():
        if row["id"] in ids:
            assert row["status"] == "unavailable"
            assert row.get("unavailable_reason")


def test_legacy_mnist_cnn_rnn_marked_unavailable_without_file():
    for cid in ("cnn-nn-visualizer", "rnn-nn-visualizer"):
        row = _meta_of(cid)
        assert row["status"] == "unavailable"
        assert row.get("unavailable_reason")


def test_invalid_model_id_raises():
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_get_and_load_error_propagation():
    adapter = registry.get("cnn-nn-visualizer")  # file missing here
    ok, reason = adapter.check_available()
    assert ok is False
    assert reason
    with pytest.raises(ModelLoadError):
        adapter.ensure_loaded()
