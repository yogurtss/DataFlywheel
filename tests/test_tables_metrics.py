import pytest

from dataflywheel.tables import canonical_html, html_to_otsl, otsl_to_html, parse_otsl
from dataflywheel.metrics import MetricEngine, score_rows, teds
from conftest import candidate, sample


def test_combined_spans_and_empty():
    html = '<table><tr><td rowspan="2" colspan="2">A &amp; B</td><td></td></tr><tr><td>C<br>D</td></tr></table>'
    expected = '<fcel>A & B<lcel><ecel><nl><ucel><xcel><fcel>C\nD<nl>'
    assert html_to_otsl(html) == expected
    assert canonical_html(html) == otsl_to_html(expected)
    assert html_to_otsl(otsl_to_html(expected)) == expected


@pytest.mark.parametrize("otsl", ["<lcel><nl>", "<ucel><nl>", "<xcel><nl>", "<fcel>a<nl><ecel><ecel><nl>",
                                    "<fcel>a<lcel><nl><ucel><ecel><nl>", "garbage", "<ecel>text<nl>"])
def test_bad_otsl(otsl):
    with pytest.raises(ValueError):
        parse_otsl(otsl)


@pytest.mark.parametrize("html", ["<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>",
                                  "<table><tr><td rowspan='3'>a</td></tr></table>",
                                  "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>"])
def test_unrepresentable_tables(html):
    with pytest.raises(ValueError):
        html_to_otsl(html)


def test_reserved_token_text():
    html = "<table><tr><td>&lt;fcel&gt; &lt;script&gt;</td></tr></table>"
    with pytest.raises(ValueError, match="delimiter"):
        html_to_otsl(html)
    html = "<table><tr><td>A &amp; B &lt;script&gt;</td></tr></table>"
    assert canonical_html(html) == otsl_to_html(html_to_otsl(html))


def test_teds_hand_computed(config):
    a = canonical_html("<table><tr><td>ab</td></tr></table>")
    b = canonical_html("<table><tr><td>ac</td></tr></table>")
    # table descendants: tbody, tr, td; one cell rename has cost 1/2.
    assert teds(a, b) == pytest.approx(1 - 0.5 / 3)
    assert teds(a, b, structure_only=True) == 1
    engine = MetricEngine(config)
    assert engine.compare("table", "garbage", a)["status"] == "invalid_prediction"
    assert engine.compare("table", a, "garbage")["score"] is None


def test_text_and_formula_metrics(config):
    e = MetricEngine(config)
    assert e.compare("text", "axc", "abc")["cer"] == pytest.approx(1 / 3)
    assert e.compare("text", "abc", "abc")["score"] == 1
    assert e.compare("text", "", "abc")["score"] == 0
    assert e.compare("text", "你好", "你好")["wer"] is None
    assert e.compare("formula", "$$x+1$$", "x+1")["score"] == 1
    assert e.compare("formula", "x", "x+1")["metric"] == "formula_edit_proxy"


def test_cdm_dependency_not_silent(config):
    config["metrics"]["formula"] = "cdm"
    engine = MetricEngine(config)
    with pytest.raises(RuntimeError, match="CDM requires"):
        engine.preflight({"formula"})
    result = engine.compare("formula", "x", "y")
    assert result["status"] == "metric_error" and result["score"] is None


def test_regression_and_missing_gt(config):
    row = sample([candidate("hello", None, "base"), candidate("hallo", None)])
    r = score_rows([row], config)[0]
    assert r["signals"]["regression"] == pytest.approx(0.2)
    assert r["signals"]["disagreement"] == pytest.approx(0.2)
    row.update(target=None, score_gt=None)
    r = score_rows([row], config)[0]
    assert r["signals"]["gt_error"] is None and r["signals"]["disagreement"] > 0


def test_truncated_response_never_scored(config):
    row = sample([candidate("hello", None, "base"), candidate("wrong", None, finish="length")])
    r = score_rows([row], config)[0]
    assert r["candidates"][1]["quality"]["score"] is None
    assert r["signals"]["disagreement"] is None
