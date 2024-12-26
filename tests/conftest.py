import pytest
from pathlib import Path


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(
            reason="Integration test - use --integration to run"
        )
        for item in items:
            test_path = Path(item.fspath)
            if "integration" in test_path.parts:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    print("Adding custom option")  # Debug print
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
