import pytest

from poker.cli import main


def test_equity_command(capsys):
    code = main(["equity", "--hero", "AhKh", "--vs", "QsQc", "--sims", "2000", "--seed", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "AhKh vs QsQc" in out
    assert "equity" in out
    assert "win" in out


def test_equity_command_with_range_and_board(capsys):
    code = main(
        ["equity", "--hero", "AsKs", "--vs", "QQ+, AKo",
         "--board", "Ah7d2c", "--sims", "1500", "--seed", "2"]
    )
    assert code == 0
    assert "on Ah7d2c" in capsys.readouterr().out


def test_pushfold_command(capsys):
    code = main(["pushfold", "--hero", "AsAh", "--stack", "20", "--sims", "1500", "--seed", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "PUSH" in out
    assert "EV(push)" in out


def test_bad_input_returns_error_code(capsys):
    code = main(["equity", "--hero", "ZZ", "--vs", "QsQc", "--sims", "10"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_missing_subcommand_exits():
    with pytest.raises(SystemExit):
        main([])
