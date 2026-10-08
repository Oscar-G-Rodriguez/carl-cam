"""Empty-VM inspection cannot promote a runtime or tolerate host integrations."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def inspector():
    path = Path(__file__).resolve().parents[1] / "tools/inspect_windows_test_vm.py"
    spec = importlib.util.spec_from_file_location("windows_vm_inspector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def info(inspector):
    return inspector.EXPECTED | {
        "name": "private", "UUID": "machine", "CfgFile": "config", "Private-SATA-0-0": "disk",
        **{f"nic{i}": "none" for i in range(1, 37)},
    }


@pytest.fixture
def xml():
    return """<VirtualBox xmlns="http://www.virtualbox.org/">
      <Machine name="private" uuid="{machine}" OSType="Windows11_64">
        <MediaRegistry><HardDisks><HardDisk uuid="{disk}" location="private-windows.vdi"
          format="VDI" type="Normal" /></HardDisks></MediaRegistry>
        <Hardware>
          <TrustedPlatformModule type="v2_0" location="" />
          <Network><Adapter slot="0" enabled="false" /></Network>
          <StorageControllers><StorageController>
            <AttachedDevice type="HardDisk" port="0" device="0" hotpluggable="false">
              <Image uuid="{disk}" />
            </AttachedDevice>
            <AttachedDevice type="DVD" port="1" device="0" passthrough="false" />
          </StorageController></StorageControllers>
        </Hardware>
      </Machine></VirtualBox>"""


def test_native_information_roundtrip_is_typed(inspector, info):
    text = "\n".join(f"{json.dumps(key)}={json.dumps(value)}" for key, value in info.items())
    parsed = inspector.parse_info(text + "\n rec_screen0\n")
    assert parsed == info
    inspector.check_info(parsed, "private", "machine", Path("config"), Path("disk"))


def test_duplicate_native_information_is_refused(inspector):
    with pytest.raises(ValueError, match="Duplicate"):
        inspector.parse_info('VMState="poweroff"\nVMState="running"')


@pytest.mark.parametrize("key", [
    "VMState", "SecureBoot", "memory", "cpus", "usb", "audio_in", "audio_out", "clipboard",
    "clipboard_file_transfers", "draganddrop", "vrde", "recording_enabled",
    "autostart-enabled", "tracing-enabled", "nic1", "nic2", "nic36", "Private-SATA-1-0",
    "CfgFile", "Private-SATA-0-0", "UUID", "name",
])
def test_changed_policy_or_target_is_refused(inspector, info, key):
    info[key] = "changed"
    with pytest.raises(ValueError, match="Unexpected or absent"):
        inspector.check_info(info, "private", "machine", Path("config"), Path("disk"))


@pytest.mark.parametrize("key", ["VMState", "SecureBoot", "nic1", "nic36", "usb", "memory"])
def test_absent_policy_is_refused(inspector, info, key):
    del info[key]
    with pytest.raises(ValueError, match="Unexpected or absent"):
        inspector.check_info(info, "private", "machine", Path("config"), Path("disk"))


def test_numeric_policy_cannot_be_string(inspector, info):
    info["memory"] = "4096"
    with pytest.raises(ValueError, match="memory"):
        inspector.check_info(info, "private", "machine", Path("config"), Path("disk"))


@pytest.mark.parametrize("key", ["SharedFolderNameMachineMapping1", "Forwarding(0)",
                                 "USBFilterName1", "storagecontrollername1"])
def test_extra_integration_is_refused(inspector, info, key):
    info[key] = "unexpected"
    with pytest.raises(ValueError, match="Unexpected sharing"):
        inspector.check_info(info, "private", "machine", Path("config"), Path("disk"))


def test_xml_proves_only_the_empty_configuration(inspector, xml):
    assert inspector.check_xml(xml, "private", "machine", "disk") == 2


@pytest.mark.parametrize("tag", ["SharedFolder", "USBFilter", "USBController", "HostDrive",
                                 "HostPci", "Snapshot", "ExtraData", "Forwarding",
                                 "BridgedInterface", "HostOnlyInterface", "HostOnlyNetwork"])
def test_forbidden_xml_integration_is_refused(inspector, xml, tag):
    xml = xml.replace("</Hardware>", f"<{tag}/></Hardware>")
    with pytest.raises(ValueError, match="Unexpected host integration"):
        inspector.check_xml(xml, "private", "machine", "disk")


@pytest.mark.parametrize("before,after", [
    ('enabled="false"', 'enabled="true"'),
    ('type="v2_0"', 'type="host"'),
    ('location=""', 'location="host"'),
    ('location="private-windows.vdi"', 'location="other.vdi"'),
    ('passthrough="false"', 'passthrough="true"'),
    ('hotpluggable="false"', 'hotpluggable="true"'),
    ('<Image uuid="{disk}" />', '<Image uuid="{other}" />'),
    ('name="private"', 'name="other"'),
])
def test_unsafe_xml_changes_are_refused(inspector, xml, before, after):
    with pytest.raises(ValueError):
        inspector.check_xml(xml.replace(before, after), "private", "machine", "disk")


def test_attached_iso_is_not_an_empty_vm(inspector, xml):
    xml = xml.replace('passthrough="false" />', 'passthrough="false"><Image uuid="{iso}" />'
                      '</AttachedDevice>')
    with pytest.raises(ValueError, match="Unexpected storage attachment"):
        inspector.check_xml(xml, "private", "machine", "disk")


def test_changed_or_missing_file_fails(inspector, tmp_path):
    with pytest.raises(ValueError):
        inspector.ordinary_file(tmp_path / "absent")
    target = tmp_path / "evidence"
    target.write_bytes(b"evidence")
    with pytest.raises(ValueError):
        inspector.ordinary_file(target, limit=1)


def test_reader_only_has_read_commands_and_no_vision_import():
    source = (Path(__file__).resolve().parents[1] / "tools/inspect_windows_test_vm.py").read_text()
    assert 'read_command("showvminfo"' in source
    assert 'read_command("showmediuminfo"' in source
    assert 'read_command("--version"' in source
    assert "startvm" not in source
    assert "import mediapipe" not in source
    assert "privacy_verified\": False" in source
