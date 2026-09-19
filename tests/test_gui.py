from pathlib import Path


def test_gui_reproducibility_script(tmp_path):
    from mixsiapy.gui import write_reproducibility_bundle

    config = {
        "mixture_file": "mixture.csv",
        "source_file": "source.csv",
        "discrimination_file": "discrimination.csv",
        "iso_names": ["d13C", "d15N"],
        "composition_formula": "species * biomass + species * diversity",
        "factors": ["spcomp"],
        "fac_random": [True],
        "fac_nested": [False],
        "cont_effects": [],
        "run": "test",
    }
    script = write_reproducibility_bundle(tmp_path, config)
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
    assert (tmp_path / "analysis_config.json").exists()
    assert (tmp_path / "environment.txt").exists()


def test_gui_app_initial_render():
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).parents[1] / "mixsiapy" / "gui" / "app.py"
    rendered = AppTest.from_file(str(app), default_timeout=30).run()
    assert not rendered.exception
    assert rendered.tabs[0].label == "01  DATA"
    assert rendered.tabs[1].label == "02  MODEL"
