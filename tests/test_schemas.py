from autoevidence.schemas.enums import Disposal, FaultType
from autoevidence.schemas.models import GoldLabel
from autoevidence.schemas.validate import assert_label_constraints
import pytest


def test_normal_requires_none_fault():
    with pytest.raises(Exception):
        GoldLabel(quality_event=False, fault_type=FaultType.BRAKE, disposal=Disposal.ARCHIVE)


def test_event_ok():
    g = GoldLabel(quality_event=True, fault_type=FaultType.BRAKE, disposal=Disposal.CRITICAL)
    assert g.fault_type == FaultType.BRAKE


def test_assert_constraints():
    assert assert_label_constraints(False, "无", "归档") == []
    assert assert_label_constraints(True, "无", "紧急复核")
