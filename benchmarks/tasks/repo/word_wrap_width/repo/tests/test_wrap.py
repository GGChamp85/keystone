from wrap import word_wrap


def test_short_text_fits_on_one_line():
    assert word_wrap("hello world", 20) == ["hello world"]


def test_wraps_at_the_given_width_without_ever_exceeding_it():
    lines = word_wrap("aa bb cc dd", 10)
    assert lines == ["aa bb cc", "dd"]
    assert all(len(line) <= 10 for line in lines)


def test_a_single_word_longer_than_width_is_its_own_line():
    assert word_wrap("supercalifragilisticexpialidocious", 10) == ["supercalifragilisticexpialidocious"]


def test_multiple_lines_are_produced_for_long_text():
    lines = word_wrap("one two three four five six seven eight", 12)
    assert all(len(line) <= 12 for line in lines)
    assert " ".join(lines) == "one two three four five six seven eight"
