from __future__ import annotations

import numpy as np
import indago


def test_optimizer_registry_consistency() -> None:
    expected_names = [opt.__name__ for opt in indago.optimizers]

    assert indago.optimizers_name_list == expected_names
    assert set(indago.optimizers_dict) == set(expected_names)
    assert all(indago.optimizers_dict[name] is opt for name, opt in zip(expected_names, indago.optimizers))
    assert indago.NelderMead is indago.NM
