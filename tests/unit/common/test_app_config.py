"""ADR-015 infrastructure config: .env.local is read through an allowlist; secrets never enter the app."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common import app_config as ac


def _env_local(root: Path, text: str) -> None:
    (root / ac.ENV_FILE).write_text(text)


def test_only_infra_keys_are_read_never_secrets(tmp_path: Path) -> None:
    secret = "hf_" + "Xy" * 17  # token shape built at run time
    _env_local(tmp_path, f'VLM_PORT=7999\nexport VLM_PROFILE="personal"\nHF_TOKEN={secret}\n# VLM_DATA_ROOT=/x\n')
    values = ac.read_env_local(tmp_path)
    assert values == {"VLM_PORT": "7999", "VLM_PROFILE": "personal"}
    assert secret not in repr(ac.load_infra(tmp_path, environ={}, policy={}))


def test_environment_wins_and_defaults_apply(tmp_path: Path) -> None:
    _env_local(tmp_path, "VLM_PORT=7999\n")
    cfg = ac.load_infra(tmp_path, environ={"VLM_PORT": "7871", "HF_TOKEN": "x"}, policy={})
    assert (cfg.port, cfg.profile, cfg.data_root) == (7871, None, None)
    assert ac.data_root(tmp_path, cfg) == tmp_path.resolve()
    assert ac.load_infra(tmp_path / "none", environ={}, policy={}).port == 7870


def test_bad_values_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        ac.load_infra(tmp_path, environ={"VLM_DATA_ROOT": "relative/dir"}, policy={})
    with pytest.raises(ValueError, match="VLM_PROFILE"):
        ac.load_infra(tmp_path, environ={"VLM_PROFILE": "root"}, policy={})


def test_a_symlinked_env_file_is_ignored(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere"
    real.write_text("VLM_PORT=7999\n")
    (tmp_path / ac.ENV_FILE).symlink_to(real)
    assert ac.read_env_local(tmp_path) == {}


def test_policy_path_ignores_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/tmp/evil")  # noqa: S108 - the point of the test
    assert str(ac.policy_path()).endswith(".config/ntc-training/policy.toml")
    assert not str(ac.policy_path()).startswith("/tmp/evil")  # noqa: S108


@pytest.mark.parametrize(
    ("admin", "local", "expected"),
    [
        ("shared_dev", "personal", "shared_dev"),  # env never relaxes the admin profile
        ("personal", "shared_dev", "shared_dev"),  # env may tighten it
        ("personal", None, "personal"),
        (None, "personal", "personal"),  # no admin policy: a personal machine decides itself
        (None, None, None),
        ("bogus", "personal", "shared_dev"),  # an unknown admin value fails closed
    ],
)
def test_admin_profile_is_the_base_and_env_only_tightens(admin: str | None, local: str | None, expected: str) -> None:
    policy = {"deployment": {"profile": admin}} if admin else {}
    assert ac.effective_profile(local, policy) == expected


def test_policy_file_is_read_safely(tmp_path: Path) -> None:
    d = tmp_path / "cfg"
    d.mkdir(mode=0o755)
    f = d / "policy.toml"
    f.write_text('[deployment]\nprofile = "shared_dev"\n')
    f.chmod(0o644)
    assert ac.read_policy(f) == {"deployment": {"profile": "shared_dev"}}
    f.chmod(0o666)
    assert ac.read_policy(f) is None  # world-writable: ignored
    f.chmod(0o644)
    link = d / "link.toml"
    link.symlink_to(f)
    assert ac.read_policy(link) is None


def test_data_root_must_be_narrow_and_owned(tmp_path: Path) -> None:
    home = ac._home()
    for bad in ("/", str(home), str(home.parent), "/etc/ntc", "/usr/local/ntc", "/proc/x"):
        with pytest.raises(ValueError, match="VLM_DATA_ROOT"):
            ac.check_data_root(bad)
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.raises(ValueError, match="writable"):
        ac.check_data_root(str(shared))
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        ac.check_data_root(str(link))
    good = tmp_path / "data"
    good.mkdir(mode=0o755)
    assert ac.check_data_root(str(good)) == good.resolve()
    assert ac.check_data_root(str(tmp_path / "not-yet")) == (tmp_path / "not-yet").resolve()
