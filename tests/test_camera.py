from chaihuo_reachy.camera import v4l2_open_candidates


def test_v4l2_open_candidates_keep_path_and_numeric_index() -> None:
    assert v4l2_open_candidates("/dev/video2") == ["/dev/video2", 2]
    assert v4l2_open_candidates(2) == [2]
