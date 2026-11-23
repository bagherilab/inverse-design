"""Tests for the generic ARCADE XML template reader/writer used by the GUI."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

import inverse_design
from inverse_design.gui.spec import ParamRef, SimulationSettings
from inverse_design.gui.xml_template import (
    TemplateError,
    default_range,
    detect_parameters,
    format_arcade_number,
    read_simulation_settings,
    write_particle_xml,
    write_template_copy,
)

SAMPLES = Path(inverse_design.__file__).parent / "sample_inputs"
COMBINED = SAMPLES / "sample_combined_v3.xml"
TCELL = SAMPLES / "T-cell.xml"
ALL_SAMPLES = sorted(SAMPLES.glob("*.xml"))

# ARCADE Series.DISTRIBUTION_REGEX and MiniBox.NUMBER_REGEX.
ARCADE_DIST = re.compile(r"^([A-Z_]+)\(([A-Z]+)=(-?[\d.]+)(?:,([A-Z]+)=(-?[\d.]+))*\)$")
ARCADE_NUMBER = re.compile(r"^(-?\d*\.\d*)$|^(-?\d+)$|^(-?\d+E-?\d+)$|^(-?\d*\.\d*E-?\d+)$")


def _by_key(path: Path) -> dict:
    return {p.ref.key: p for p in detect_parameters(path)}


def _write(tmp_path: Path, text: str, name: str = "t.xml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def test_detect_combined_template():
    params = _by_key(COMBINED)

    mu = params["population:cancerous:CELL_VOLUME.MU"]
    sigma = params["population:cancerous:CELL_VOLUME.SIGMA"]
    assert (mu.xml_value, mu.label) == (2250.0, "CELL_VOLUME_MU")
    assert (sigma.xml_value, sigma.label) == (200.0, "CELL_VOLUME_SIGMA")
    assert mu.raw_value == "NORMAL(MU=2250,SIGMA=200)"

    nf = params["population:cancerous:NECROTIC_FRACTION"]
    assert (nf.xml_value, nf.label, nf.ref.field) == (0.5, "NECROTIC_FRACTION", None)

    glu = params["population:cancerous:metabolism/GLUCOSE_UPTAKE_RATE.MU"]
    assert glu.label == "GLUCOSE_UPTAKE_RATE_MU"
    assert glu.xml_value == pytest.approx(1.12)

    assert params["population:cancerous:CELL_AGE.MIN"].ref.field == "MIN"
    assert params["population:cancerous:CELL_AGE.MAX"].ref.field == "MAX"

    gen = params["layer:GLUCOSE:CONCENTRATION@generator"]
    assert (gen.xml_value, gen.label) == (0.005, "GLUCOSE_CONCENTRATION_GENERATOR")
    assert gen.ref == ParamRef("layer", "GLUCOSE", "CONCENTRATION", "generator", None)
    assert params["layer:OXYGEN:INITIAL_CONCENTRATION"].label == "OXYGEN_INITIAL_CONCENTRATION"
    assert params["layer:TGFA:DIFFUSIVITY@diffuser"].xml_value == 51.8

    # Non-numeric values are skipped.
    ids = {p.ref.param_id for p in params.values()}
    assert "GEOMETRY" not in ids  # "hex"
    assert "X_SPACING" not in ids and "Y_SPACING" not in ids  # "*:2"
    # 15 distributions x 2 fields + 4 plain population numbers + 8 layer numbers
    assert len(params) == 42


def test_detect_document_order():
    labels = [p.label for p in detect_parameters(COMBINED)]
    assert labels[:3] == ["CELL_VOLUME_MU", "CELL_VOLUME_SIGMA", "APOPTOSIS_AGE_MU"]
    assert labels[-1] == "TGFA_DIFFUSIVITY_DIFFUSER"


@pytest.mark.parametrize("path", ALL_SAMPLES, ids=lambda p: p.name)
def test_labels_unique_and_keys_round_trip(path: Path):
    params = detect_parameters(path)
    assert params
    labels = [p.label for p in params]
    keys = [p.ref.key for p in params]
    assert len(set(labels)) == len(labels)
    assert len(set(keys)) == len(keys)
    for p in params:
        assert re.fullmatch(r"[A-Z_][A-Z0-9_]*", p.label), p.label
        assert ParamRef.from_key(p.ref.key) == p.ref


def test_labels_disambiguate_with_owner_id():
    params = _by_key(TCELL)
    assert params["population:healthy:CAR_ANTIGENS.MU"].label == "HEALTHY_CAR_ANTIGENS_MU"
    assert params["population:cancerous:CAR_ANTIGENS.MU"].label == "CANCEROUS_CAR_ANTIGENS_MU"
    # Numeric owner ids get the owner tag so labels never start with a digit.
    assert params["population:8:CELL_VOLUME.MU"].label == "POPULATION_8_CELL_VOLUME_MU"
    assert params["population:4:CELL_VOLUME.MU"].label == "POPULATION_4_CELL_VOLUME_MU"
    # Unique short names stay short.
    assert params["action:ADD_T_CELLS:DOSE"].label == "DOSE"
    # Scientific notation accepted by ARCADE for plain numbers.
    assert params["population:8:CAR_AFFINITY"].xml_value == pytest.approx(1e-6)
    # Parameters given by scale (no value attribute) are skipped.
    assert "population:cancerous:METABOLIC_PREFERENCE" not in params


def test_labels_disambiguate_same_name_in_one_owner(tmp_path: Path):
    path = _write(
        tmp_path,
        """<set><series name="s">
        <agents><populations><population id="a">
            <population.parameter id="metabolism/RATE" value="1" />
            <population.parameter id="signaling/RATE" value="2" />
        </population></populations></agents>
        <environments><layers><layer id="X-Y">
            <layer.parameter id="D" operation="diffuser" value="3" />
        </layer></layers></environments>
        </series></set>""",
    )
    params = _by_key(path)
    assert params["population:a:metabolism/RATE"].label == "A_METABOLISM_RATE"
    assert params["population:a:signaling/RATE"].label == "A_SIGNALING_RATE"
    assert params["layer:X-Y:D@diffuser"].label == "X_Y_D_DIFFUSER"


def test_detect_skips_non_arcade_values(tmp_path: Path):
    path = _write(
        tmp_path,
        """<set><series name="s"><patch>
            <patch.parameter id="A" value="1/2" />
            <patch.parameter id="B" value="NORMAL(MU=1e-5,SIGMA=1)" />
            <patch.parameter id="C" value="NORMAL(MU=1, SIGMA=1)" />
            <patch.parameter id="D" value="BERNOULLI(PROBABILITY=0.5)" />
            <patch.parameter id="E" value="TRUNCATED_NORMAL(MU=4,SIGMA=1)" />
            <patch.parameter id="F" value="-3" />
            <patch.parameter id="G" value="." />
        </patch></series></set>""",
    )
    params = _by_key(path)
    assert sorted(params) == ["::E.MU", "::E.SIGMA", "::F"]
    assert params["::F"].label == "F"
    assert params["::F"].ref.owner_tag == ""


def test_duplicate_parameter_reports_last_and_writes_all(tmp_path: Path):
    path = _write(
        tmp_path,
        """<set><series name="s"><agents><populations><population id="a">
            <population.parameter id="X" value="1" />
            <population.parameter id="X" value="2" />
        </population></populations></agents></series></set>""",
    )
    params = detect_parameters(path)
    assert len(params) == 1 and params[0].xml_value == 2.0
    out = tmp_path / "out.xml"
    write_particle_xml(path, out, {"population:a:X": 5}, read_simulation_settings(path))
    assert out.read_text().count('value="5"') == 2


# --------------------------------------------------------------------------- #
# Simulation settings
# --------------------------------------------------------------------------- #


def test_read_simulation_settings():
    assert read_simulation_settings(COMBINED) == SimulationSettings(
        seeds=10, ticks=10080, interval=720, radius=10, margin=2, seed_start=0
    )
    necrotic = read_simulation_settings(SAMPLES / "test_necrotic.xml")
    assert (necrotic.seed_start, necrotic.seeds, necrotic.radius) == (5, 1, 20)


def test_series_defaults_follow_arcade(tmp_path: Path):
    path = _write(tmp_path, '<set><series name="s" ticks="200" /></set>')
    assert read_simulation_settings(path) == SimulationSettings(
        seeds=1, ticks=200, interval=50, radius=1, margin=0, seed_start=0
    )


def test_series_settings_write(tmp_path: Path):
    sim = SimulationSettings(seeds=3, ticks=1440, interval=60, radius=6, margin=4, seed_start=7)
    out = tmp_path / "template.xml"
    write_template_copy(COMBINED, out, sim)
    assert read_simulation_settings(out) == sim
    series = re.search(r"<series [^>]*>", out.read_text()).group(0)
    assert 'start="7"' in series and 'end="9"' in series
    assert 'name="grid"' in series and 'depth="1"' in series
    # Attribute order of the template is kept.
    assert series.index("name=") < series.index("start=") < series.index("ticks=")
    # Parameters unchanged.
    assert _by_key(out) == _by_key(COMBINED)


def test_series_settings_added_when_missing(tmp_path: Path):
    path = _write(tmp_path, '<set><series name="s" /></set>')
    sim = SimulationSettings(seeds=2, ticks=10, interval=5, radius=4, margin=2)
    out = tmp_path / "out.xml"
    write_template_copy(path, out, sim)
    assert read_simulation_settings(out) == sim


@pytest.mark.parametrize("change", [{"seeds": 0}, {"ticks": 0}, {"margin": -1}, {"radius": 2.5}])
def test_invalid_simulation_settings(tmp_path: Path, change: dict):
    sim = replace(read_simulation_settings(COMBINED), **change)
    with pytest.raises(ValueError):
        write_template_copy(COMBINED, tmp_path / "out.xml", sim)


# --------------------------------------------------------------------------- #
# Writing particles
# --------------------------------------------------------------------------- #


def test_write_round_trip(tmp_path: Path):
    sim = read_simulation_settings(COMBINED)
    new_sim = replace(sim, seeds=4, ticks=2880)
    changes = {
        "population:cancerous:CELL_VOLUME.MU": 3000.0,
        "population:cancerous:metabolism/GLUCOSE_UPTAKE_RATE.SIGMA": 0.123456789,
        "population:cancerous:NECROTIC_FRACTION": 0.25,
        "population:cancerous:CELL_AGE.MAX": 100.0,
        "layer:GLUCOSE:CONCENTRATION@generator": 0.0075,
        "layer:OXYGEN:DIFFUSIVITY@diffuser": 1234.5,
    }
    out = tmp_path / "input_0.xml"
    write_particle_xml(COMBINED, out, changes, new_sim)

    assert read_simulation_settings(out) == new_sim
    before, after = _by_key(COMBINED), _by_key(out)
    assert list(after) == list(before)
    for key, param in after.items():
        assert param.label == before[key].label
        if key in changes:
            assert param.xml_value == pytest.approx(changes[key], rel=1e-9)
        else:
            assert param.xml_value == before[key].xml_value
            if param.ref.field is None:
                assert param.raw_value == before[key].raw_value


def test_write_only_named_distribution_field(tmp_path: Path):
    out = tmp_path / "out.xml"
    write_particle_xml(
        COMBINED,
        out,
        {"population:cancerous:CELL_VOLUME.MU": 3000},
        read_simulation_settings(COMBINED),
    )
    params = _by_key(out)
    assert params["population:cancerous:CELL_VOLUME.MU"].raw_value == "NORMAL(MU=3000,SIGMA=200)"

    write_particle_xml(
        COMBINED,
        out,
        {"population:cancerous:CELL_VOLUME.SIGMA": 150.5},
        read_simulation_settings(COMBINED),
    )
    params = _by_key(out)
    assert params["population:cancerous:CELL_VOLUME.MU"].raw_value == "NORMAL(MU=2250,SIGMA=150.5)"


def test_write_keeps_other_attributes_comments_and_declaration(tmp_path: Path):
    path = _write(
        tmp_path,
        """<?xml version="1.0" ?>
<!-- header comment -->
<set prefix="p_">
    <series name="s" start="0" end="0" ticks="10" interval="5" radius="4" margin="2">
        <!-- inner comment -->
        <environments><layers><layer id="GLUCOSE">
            <layer.parameter id="CONCENTRATION" operation="generator" value="0.005" units="u" />
        </layer></layers></environments>
    </series>
</set>
""",
    )
    out = tmp_path / "out.xml"
    write_particle_xml(
        path, out, {"layer:GLUCOSE:CONCENTRATION@generator": 0.01}, read_simulation_settings(path)
    )
    text = out.read_text()
    assert text.startswith('<?xml version="1.0" ?>\n<!-- header comment -->\n<set prefix="p_">')
    assert "<!-- inner comment -->" in text
    assert (
        '<layer.parameter id="CONCENTRATION" operation="generator" value="0.01" units="u" />'
        in text
    )
    assert text.endswith("</set>\n")


def test_written_numbers_are_arcade_parsable(tmp_path: Path):
    params = detect_parameters(COMBINED)
    values = {p.ref.key: p.xml_value * 1.000123e-3 for p in params}
    out = tmp_path / "out.xml"
    write_particle_xml(COMBINED, out, values, read_simulation_settings(COMBINED))
    for p in detect_parameters(out):
        raw = p.raw_value
        assert "e" not in raw.lower().replace("e(", "")  # no scientific notation
        assert ARCADE_DIST.match(raw) if p.ref.field else ARCADE_NUMBER.match(raw)
        assert p.xml_value == pytest.approx(values[p.ref.key], rel=1e-9)


def test_write_unknown_key_raises(tmp_path: Path):
    sim = read_simulation_settings(COMBINED)
    for key in (
        "population:cancerous:NOT_A_PARAM",
        "population:cancerous:NECROTIC_FRACTION.MU",  # plain number has no MU
        "population:cancerous:CELL_VOLUME",  # distribution needs a field
        "patch::GEOMETRY",  # non-numeric
    ):
        with pytest.raises(KeyError):
            write_particle_xml(COMBINED, tmp_path / "out.xml", {key: 1.0}, sim)
    assert not (tmp_path / "out.xml").exists()


def test_write_rejects_non_finite(tmp_path: Path):
    with pytest.raises(ValueError):
        write_particle_xml(
            COMBINED,
            tmp_path / "out.xml",
            {"population:cancerous:ACCURACY": float("nan")},
            read_simulation_settings(COMBINED),
        )


@pytest.mark.parametrize(
    "value, expected",
    [
        (2250.0, "2250"),
        (0.0, "0"),
        (-0.0, "0"),
        (8e-05, "0.00008"),
        (-1.5, "-1.5"),
        (1e20, "100000000000000000000"),
        (0.1 + 0.2, "0.3"),
        (1.23456789012345e-7, "0.000000123456789"),
    ],
)
def test_format_arcade_number(value: float, expected: str):
    assert format_arcade_number(value) == expected


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_template_errors(tmp_path: Path):
    cases = {
        "missing": None,
        "invalid": "<set><series name='s'></set>",
        "none": "<set><other /></set>",
        "many": "<set><series name='a' /><series name='b' /></set>",
        "bad_end": "<set><series name='s' start='3' end='1' /></set>",
        "bad_number": "<set><series name='s' ticks='lots' /></set>",
    }
    messages = {}
    for name, text in cases.items():
        path = tmp_path / f"{name}.xml"
        if text is not None:
            path.write_text(text)
        with pytest.raises(TemplateError) as info:
            read_simulation_settings(path)
        messages[name] = str(info.value)
    assert "Cannot read" in messages["missing"]
    assert "not valid XML" in messages["invalid"]
    assert "no <series>" in messages["none"]
    assert "2 <series>" in messages["many"]
    with pytest.raises(TemplateError):
        detect_parameters(tmp_path / "many.xml")
    assert issubclass(TemplateError, ValueError)


# --------------------------------------------------------------------------- #
# Default prior range
# --------------------------------------------------------------------------- #


def _ref(param_id: str, field=None) -> ParamRef:
    return ParamRef("population", "a", param_id, "", field)


@pytest.mark.parametrize(
    "value, ref, expected",
    [
        (2250.0, _ref("CELL_VOLUME", "MU"), (1125.0, 3375.0)),
        (0.0, _ref("CELL_AGE", "MIN"), (0.0, 1.0)),
        (-2.0, _ref("COMPRESSION_TOLERANCE"), (-3.0, -1.0)),
        (0.8, _ref("ACCURACY"), (0.4, 1.0)),
        (0.5, _ref("NECROTIC_FRACTION"), (0.25, 0.75)),
        (0.9, _ref("metabolism/METABOLIC_PREFERENCE", "MU"), (0.45, 1.0)),
        (0.7, _ref("metabolism/RATIO_GLUCOSE_PYRUVATE", "MU"), (0.35, 1.0)),
        (1.0, _ref("metabolism/CONVERSION_FRACTION", "MU"), (0.5, 1.0)),
        (0.8, _ref("LACTATE_RATE"), (0.4, 1.2)),  # not a fraction
        (4.0, _ref("AFFINITY_SCALE"), (2.0, 6.0)),  # fraction hint but value > 1
    ],
)
def test_default_range(value: float, ref: ParamRef, expected: tuple[float, float]):
    low, high = default_range(value, ref)
    assert (low, high) == pytest.approx(expected)
    assert low < high
