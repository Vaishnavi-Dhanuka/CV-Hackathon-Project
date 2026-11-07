from app.components.sidebar import PIPELINE_MODES


def test_lumina_exposes_original_and_all_three_restoration_modes():
    assert PIPELINE_MODES == (
        "Original Image",
        "Classical CV",
        "Official RetinexFormer",
        "Hybrid Intelligence",
    )
