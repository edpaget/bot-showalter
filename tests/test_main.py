from fantasy_baseball_manager.main import main


def test_main_importable() -> None:
    assert callable(main)
