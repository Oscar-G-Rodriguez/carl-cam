"""The installed-guest DVD check remains read-only and refuses policy drift."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def inspector(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("vision_test_vm_inspector", tools / "inspect_vision_test_vm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def info(inspector, tmp_path):
    return inspector.EXPECTED | {
        "name": inspector.NAME, "UUID": inspector.UUID, "hardwareuuid": inspector.UUID,
        "VMState": "poweroff", "cpus": 3,
        "CfgFile": str(tmp_path / inspector.VM_DIRECTORY / (inspector.NAME + ".vbox")),
        "Private-SATA-0-0": str(tmp_path / inspector.VM_DIRECTORY / "private-windows.vdi"),
        "Private-SATA-ImageUUID-0-0": inspector.DISK_UUID,
        "Private-SATA-1-0": str(tmp_path / inspector.CHECKS_DIRECTORY / "carl-cam-vm-checks.iso"),
        "Private-SATA-ImageUUID-1-0": "11111111-2222-3333-4444-555555555555",
        "Private-SATA-IsEjected-1-0": "off", "Private-SATA-tempeject-1-0": "off",
        "storagecontrollerinstance0": "0", **{f"nic{i}": "none" for i in range(1, 37)},
    }


@pytest.fixture
def xml(inspector, info):
    return f"""<VirtualBox><Machine name="{inspector.NAME}" uuid="{{{inspector.UUID}}}" OSType="Windows11_64">
    <MediaRegistry><HardDisks><HardDisk uuid="{{{inspector.DISK_UUID}}}" location="private-windows.vdi"
    format="VDI" type="Normal" /></HardDisks><DVDImages><Image uuid="{{{info['Private-SATA-ImageUUID-1-0']}}}"
    location="{info['Private-SATA-1-0']}" /></DVDImages></MediaRegistry>
    <ExtraData><ExtraDataItem name="GUI/LastNormalWindowPosition" value="-20,30,819,668" /></ExtraData>
    <Hardware><TrustedPlatformModule type="v2_0" location="" /><CPU count="3" /><Memory RAMSize="4096" />
    <Firmware type="EFI" /><Chipset type="ICH9" /><Network><Adapter slot="0" enabled="false" /></Network>
    <StorageControllers><StorageController name="Private-SATA" type="AHCI" PortCount="2">
    <AttachedDevice type="HardDisk" port="0" device="0" hotpluggable="false"><Image uuid="{{{inspector.DISK_UUID}}}" /></AttachedDevice>
    <AttachedDevice type="DVD" port="1" device="0" hotpluggable="false" passthrough="false"><Image uuid="{{{info['Private-SATA-ImageUUID-1-0']}}}" /></AttachedDevice>
    </StorageController></StorageControllers></Hardware></Machine></VirtualBox>"""


def test_explicit_disc_and_state_inspection(inspector, info, xml, tmp_path):
    dvd = Path(info["Private-SATA-1-0"])
    identity = inspector.check_information(info, tmp_path, dvd, "poweroff")
    inspector.check_configuration(xml, dvd, identity)
    info["VMState"] = "running"
    inspector.check_information(info, tmp_path, dvd, "running")
    with pytest.raises(ValueError, match="VMState"):
        inspector.check_information(info, tmp_path, dvd, "poweroff")


@pytest.mark.parametrize("key,value", [
    ("nic1", "nat"), ("nic36", "nat"), ("cpus", "3"), ("memory", "4096"),
    ("UUID", "other"), ("hardwareuuid", "other"), ("CfgFile", "other"), ("Private-SATA-0-0", "other"),
    ("Private-SATA-ImageUUID-0-0", "other"), ("Private-SATA-1-0", "other"),
    ("Private-SATA-IsEjected-1-0", "on"), ("Private-SATA-tempeject-1-0", "on"),
    ("clipboard", "bidirectional"), ("clipboard_file_transfers", "on"),
    ("usb", "on"), ("audio_in", "on"), ("recording_enabled", "on"),
    ("SecureBoot", "off"), ("Private-SATA-ImageUUID-1-0", "not-a-uuid"),
    ("storagecontrollerinstance0", 0),
])
def test_policy_drift_fails(inspector, info, tmp_path, key, value):
    dvd = Path(info["Private-SATA-1-0"])
    info[key] = value
    with pytest.raises(ValueError):
        inspector.check_information(info, tmp_path, dvd, "poweroff")


@pytest.mark.parametrize("key", ["nic36", "cpus", "Private-SATA-IsEjected-1-0"])
def test_absent_policy_fails(inspector, info, tmp_path, key):
    dvd = Path(info["Private-SATA-1-0"])
    del info[key]
    with pytest.raises(ValueError):
        inspector.check_information(info, tmp_path, dvd, "poweroff")


@pytest.mark.parametrize("key", ["SharedFolderNameMachineMapping1", "Forwarding(0)",
                                  "pcidevice0", "storagecontrollername1", "Private-SATA-2-0"])
def test_extra_integration_fails(inspector, info, tmp_path, key):
    info[key] = "unreviewed"
    with pytest.raises(ValueError, match="integration"):
        inspector.check_information(info, tmp_path, Path(info["Private-SATA-1-0"]), "poweroff")


@pytest.mark.parametrize("before,after", [
    ('enabled="false"', 'enabled="true"'), ('count="3"', 'count="2"'),
    ('RAMSize="4096"', 'RAMSize="2048"'), ('type="v2_0"', 'type="host"'),
    ('passthrough="false"', 'passthrough="true"'),
    ('GUI/LastNormalWindowPosition', 'VBoxInternal/Devices/unreviewed'),
    ('value="-20,30,819,668"', 'value="unreviewed"'),
    ('PortCount="2"', 'PortCount="3"'),
    ('</Hardware>', '<SharedFolder /></Hardware>'),
    ('</Hardware>', '<Snapshot /></Hardware>'),
    ('</StorageController>', '<AttachedDevice type="DVD" /></StorageController>'),
])
def test_xml_drift_fails(inspector, info, xml, before, after):
    with pytest.raises(ValueError):
        inspector.check_configuration(xml.replace(before, after), Path(info["Private-SATA-1-0"]),
                                      info["Private-SATA-ImageUUID-1-0"])


def test_running_diagnostics_and_duplicate_fields(inspector):
    assert inspector.parse_information('VideoMode="0,0,0"@0,0 1\nVRDEClients==0\ncpus=3\n')['cpus'] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        inspector.parse_information('cpus=3\ncpus=3')


def test_complete_read_only_flow_does_not_promote_runtime(inspector, info, xml, tmp_path, monkeypatch):
    import json

    config = Path(info["CfgFile"])
    config.parent.mkdir(parents=True)
    config.write_text(xml)
    (config.parent / "private-windows.vdi").write_bytes(b"guest fixture")
    dvd = Path(info["Private-SATA-1-0"])
    dvd.parent.mkdir(parents=True)
    dvd.write_bytes(b"disc fixture")
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments[1:])
        assert kwargs["check"] is True
        result = "7.2.20r175154" if arguments[1:] == ["--version"] else "\n".join(
            f"{json.dumps(k)}={json.dumps(v)}" for k, v in info.items()
        )
        return SimpleNamespace(stdout=result)

    monkeypatch.setattr(inspector.subprocess, "run", run)
    monkeypatch.setattr(inspector, "file_hash", lambda p: inspector.MANAGER_HASH if p == inspector.MANAGER
                        else inspector.CHECKS_HASH)
    result = inspector.inspect(tmp_path, "checks", "poweroff")
    assert calls == [["--version"], ["showvminfo", inspector.UUID, "--machinereadable"],
                     ["showvminfo", inspector.UUID, "--machinereadable"]]
    assert result["configuration_reviewed"] is True
    for key in ("vm_started_by_check", "media_attached_by_check", "guest_files_verified",
                "windows_license_verified", "vision_runtime_loaded", "privacy_verified", "gpu_benchmarked"):
        assert result[key] is False


def test_media_bytes_fail_before_manager_execution(inspector, tmp_path, monkeypatch):
    monkeypatch.setattr(inspector, "file_hash", lambda _: "0" * 64)
    monkeypatch.setattr(inspector.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected manager call"))
    with pytest.raises(ValueError, match="bytes changed"):
        inspector.inspect(tmp_path, "checks", "poweroff")


def retained_installation_image(inspector, xml, identity=None, location=None):
    identity = inspector.WINDOWS_DVD_UUID if identity is None else identity
    location = inspector.WINDOWS_ISO if location is None else location
    return xml.replace("</DVDImages>", f'<Image uuid="{{{identity}}}" location="{location}" /></DVDImages>')


def test_exact_inactive_installation_image_is_not_an_attachment(inspector, info, xml):
    inspector.check_configuration(retained_installation_image(inspector, xml),
                                  Path(info["Private-SATA-1-0"]), info["Private-SATA-ImageUUID-1-0"])


@pytest.mark.parametrize("identity,location", [
    ("unknown", None), (None, "unknown.iso"),
    ("11111111-2222-3333-4444-555555555555", None),
])
def test_unknown_or_duplicate_inactive_image_is_refused(inspector, info, xml, identity, location):
    with pytest.raises(ValueError, match="DVD registry"):
        inspector.check_configuration(retained_installation_image(inspector, xml, identity, location),
                                      Path(info["Private-SATA-1-0"]), info["Private-SATA-ImageUUID-1-0"])


def test_third_registered_image_is_refused(inspector, info, xml):
    extra = retained_installation_image(inspector, retained_installation_image(inspector, xml))
    with pytest.raises(ValueError, match="DVD registry"):
        inspector.check_configuration(extra, Path(info["Private-SATA-1-0"]),
                                      info["Private-SATA-ImageUUID-1-0"])


def observer_registry(inspector, info, xml, root):
    old = info["Private-SATA-1-0"]
    dvd = root / inspector.OBSERVER_DIRECTORY / "carl-cam-vm-checks.iso"
    result = retained_installation_image(inspector, xml.replace(old, str(dvd)))
    previous = root / inspector.CHECKS_DIRECTORY / "carl-cam-vm-checks.iso"
    result = result.replace("</DVDImages>",
        f'<Image uuid="{{{inspector.CHECKS_DVD_UUID}}}" location="{previous}" /></DVDImages>')
    return dvd, result


def test_observer_permits_only_exact_reviewed_previous_images(inspector, info, xml, tmp_path):
    dvd, data = observer_registry(inspector, info, xml, tmp_path)
    inspector.check_configuration(data, dvd, info["Private-SATA-ImageUUID-1-0"])


@pytest.mark.parametrize("case", ["old_uuid", "old_path", "fourth", "duplicate", "hardware_uuid"])
def test_observer_registry_or_machine_identity_drift_refused(inspector, info, xml, tmp_path, case):
    dvd, data = observer_registry(inspector, info, xml, tmp_path)
    if case == "old_uuid":
        data = data.replace(inspector.CHECKS_DVD_UUID, "unknown")
    elif case == "old_path":
        data = data.replace(str(tmp_path / inspector.CHECKS_DIRECTORY), str(tmp_path / "other"))
    elif case == "fourth":
        data = data.replace("</DVDImages>", '<Image uuid="{unknown}" location="other.iso" /></DVDImages>')
    elif case == "duplicate":
        data = data.replace(inspector.CHECKS_DVD_UUID, info["Private-SATA-ImageUUID-1-0"])
    else:
        data = data.replace('OSType="Windows11_64"', 'OSType="Windows11_64" hardwareUUID="{unknown}"')
    with pytest.raises(ValueError):
        inspector.check_configuration(data, dvd, info["Private-SATA-ImageUUID-1-0"])


def test_old_disc_mode_cannot_accept_the_observer_registry(inspector, info, xml, tmp_path):
    dvd, data = observer_registry(inspector, info, xml, tmp_path)
    old = Path(info["Private-SATA-1-0"])
    with pytest.raises(ValueError):
        inspector.check_configuration(data.replace(str(dvd), str(old)), old,
                                      info["Private-SATA-ImageUUID-1-0"])


def test_retained_image_cannot_replace_the_selected_attachment(inspector, info, xml):
    extra = retained_installation_image(inspector, xml)
    extra = extra.replace('port="1" device="0" hotpluggable="false" passthrough="false"><Image uuid="{' +
                          info["Private-SATA-ImageUUID-1-0"] + '}"',
                          'port="1" device="0" hotpluggable="false" passthrough="false"><Image uuid="{' +
                          inspector.WINDOWS_DVD_UUID + '}"')
    with pytest.raises(ValueError, match="storage attachment"):
        inspector.check_configuration(extra, Path(info["Private-SATA-1-0"]),
                                      info["Private-SATA-ImageUUID-1-0"])


def test_observed_maximized_window_metadata_is_allowed(inspector, info, xml):
    maximized = xml.replace('value="-20,30,819,668"', 'value="-2560,23,2048,1129,max"')
    inspector.check_configuration(maximized, Path(info["Private-SATA-1-0"]),
                                  info["Private-SATA-ImageUUID-1-0"])


@pytest.mark.parametrize("value", ["-2560,23,2048,1129,min", "-2560,23,2048,1129,max,extra",
                                  "-2560,23,2048,1129,MAX", "-2560,23,2048,1129,"])
def test_unrecognized_window_metadata_is_refused(inspector, info, xml, value):
    changed = xml.replace('value="-20,30,819,668"', f'value="{value}"')
    with pytest.raises(ValueError, match="extra-data"):
        inspector.check_configuration(changed, Path(info["Private-SATA-1-0"]),
                                      info["Private-SATA-ImageUUID-1-0"])
