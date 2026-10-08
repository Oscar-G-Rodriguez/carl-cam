"""Exercise the actual PowerShell read policy using generated VM metadata only.

The extracted reader gets a stub manager that rejects every command except one
exact showvminfo read. No guest is started and no inference library is imported.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/start_private_windows_vm.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is required")


@pytest.fixture
def info():
    return {
        "UUID": "fixture-machine", "name": "fixture-private", "CfgFile": "fixture-config",
        "memory": 4096, "cpus": 3, "audio": "none", "SecureBoot": "on",
        "storagecontrollername0": "Private-SATA", "Private-SATA-0-0": "fixture-disk",
        "clipboard": "disabled", "draganddrop": "disabled",
        **{f"nic{number}": "none" for number in range(1, 37)},
        **dict.fromkeys(("usb", "ehci", "xhci", "audio_in", "audio_out", "vrde",
                         "recording_enabled", "recording_started", "clipboard_file_transfers",
                         "autostart-enabled", "tracing-enabled", "tracing-allow-vm-access"), "off"),
    }


def quote_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def run_reader(tmp_path, info, expected_cpus=3, suffix=""):
    fixture = tmp_path / "generated-vm-info.txt"
    fixture.write_text("\n".join(f"{key}={json.dumps(value)}" for key, value in info.items())
                       + suffix, encoding="utf-8")
    command = f"""$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile({quote_path(SCRIPT)}, [ref]$tokens, [ref]$errors)
if ($errors.Count) {{ throw 'Source script syntax failed.' }}
$reader = @($ast.FindAll({{param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Read-PrivateConfiguration'}}, $true))
if ($reader.Count -ne 1) {{ throw 'Expected exactly one actual reader definition.' }}
$fixtureText = Get-Content -LiteralPath {quote_path(fixture)} -Raw
$machineUuid = 'fixture-machine'; $expectedName = 'fixture-private'
$expectedConfiguration = 'fixture-config'; $expectedDisk = 'fixture-disk'
$ExpectedCpuCount = {expected_cpus}
function Invoke-PrivateManager([string[]]$CommandArguments) {{
  if (($CommandArguments -join '|') -cne 'showvminfo|fixture-machine|--machinereadable') {{ throw 'Mutation or unexpected manager call forbidden.' }}
  return $fixtureText
}}
Invoke-Expression $reader[0].Extent.Text
try {{
  $result = Read-PrivateConfiguration
  @{{accepted=$true; cpu_count=$result.cpus; diagnostics=$result.VideoMode}} | ConvertTo-Json -Compress
}} catch {{
  @{{accepted=$false; error=$_.Exception.Message}} | ConvertTo-Json -Compress
}}
"""
    result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8", timeout=20, check=True)
    return json.loads(result.stdout.strip())


@pytest.mark.parametrize("cpu_count", (2, 3))
def test_explicit_matching_cpu_count_passes(tmp_path, info, cpu_count):
    info["cpus"] = cpu_count
    result = run_reader(tmp_path, info, cpu_count)
    assert result["accepted"] and result["cpu_count"] == cpu_count


@pytest.mark.parametrize("actual,expected", ((2, 3), (3, 2), (4, 3), ("3", 3), (True, 3)))
def test_changed_or_coerced_cpu_count_fails(tmp_path, info, actual, expected):
    info["cpus"] = actual
    assert not run_reader(tmp_path, info, expected)["accepted"]


@pytest.mark.parametrize("key,value", (
    ("memory", "4096"), ("UUID", "another-vm"), ("name", "another-vm"),
    ("CfgFile", "another-config"), ("Private-SATA-0-0", "another-disk"),
    ("SecureBoot", "off"), ("nic1", "nat"), ("nic36", "hostonly"),
    ("usb", "on"), ("clipboard", "bidirectional"), ("tracing-enabled", "on"),
    ("sharedfoldername1", "personal"), ("storagecontrollername1", "additional"),
))
def test_policy_and_target_changes_fail(tmp_path, info, key, value):
    info[key] = value
    assert not run_reader(tmp_path, info)["accepted"]


@pytest.mark.parametrize("key", ("cpus", "memory", "UUID", "CfgFile", "nic36", "usb"))
def test_missing_policy_fails(tmp_path, info, key):
    del info[key]
    assert not run_reader(tmp_path, info)["accepted"]


def test_known_live_diagnostics_do_not_bypass_policy(tmp_path, info):
    suffix = '\nVideoMode="1024,768,32"@0,0 1\nVRDEClients==0\n rec_screen0\n'
    result = run_reader(tmp_path, info, suffix=suffix)
    assert result["accepted"] and result["diagnostics"] == '"1024,768,32"@0,0 1'
    info["nic36"] = "nat"
    assert not run_reader(tmp_path, info, suffix=suffix)["accepted"]


def test_duplicate_policy_fails(tmp_path, info):
    result = run_reader(tmp_path, info, suffix='\ncpus=3\n')
    assert not result["accepted"] and "Duplicate" in result["error"]


@pytest.mark.parametrize("arguments,expected_message", (
    (["-ExpectedCpuCount", "4"], "ValidateSet"),
    (["-Start", "-ExpectedCpuCount", "3"], "Initial empty-guest startup"),
    (["-Start", "-CheckRunning"], "Choose startup"),
))
def test_invalid_invocation_stops_before_manager_or_iso_access(arguments, expected_message):
    result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-File", str(SCRIPT),
                             "-IsoPath", "missing-fixture-never-accessed.iso", *arguments],
                            capture_output=True, text=True, encoding="utf-8", timeout=20, check=False)
    assert result.returncode != 0
    assert expected_message in result.stderr
    assert "missing-fixture-never-accessed.iso" not in result.stdout
