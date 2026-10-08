"""Read-only audit of the empty private VM; never starts a guest or loads vision.

This gate proves only the present powered-off configuration. It cannot establish
boot compatibility, a Windows license, live guest isolation or runtime privacy.
"""

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

MANAGER = Path(r"C:\Program Files\Oracle\VirtualBox\VBoxManage.exe")
MANAGER_HASH = "6852390a57d454d3b7c9ae27e66e45c367699faf2807b450de084b1f7d0203c1"
EXPECTED = {
    "platformArchitecture": "x86", "ostype": "Windows 11 (64-bit)",
    "VMState": "poweroff", "memory": 4096, "cpus": 2, "firmware": "EFI",
    "SecureBoot": "on", "chipset": "ich9", "ioapic": "on",
    "graphicscontroller": "vboxsvga", "accelerate3d": "off", "vram": 128,
    "boot1": "dvd", "boot2": "disk", "boot3": "none", "boot4": "none",
    "usb": "off", "ehci": "off", "xhci": "off", "audio": "none",
    "audio_in": "off", "audio_out": "off", "clipboard": "disabled",
    "clipboard_file_transfers": "off", "draganddrop": "disabled", "vrde": "off",
    "recording_enabled": "off", "recording_started": "off", "autostart-enabled": "off",
    "teleporterenabled": "off", "tracing-enabled": "off", "tracing-allow-vm-access": "off",
    "nic1": "none", "hidpointing": "ps2mouse",
    "hidkeyboard": "ps2kbd", "uart1": "off", "uart2": "off", "uart3": "off",
    "uart4": "off", "lpt1": "off", "lpt2": "off",
    "storagecontrollername0": "Private-SATA", "storagecontrollertype0": "IntelAhci",
    "storagecontrollerportcount0": "2", "Private-SATA-1-0": "emptydrive",
}


def ordinary_file(path, limit=None):
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.is_symlink() or parent.is_junction():
            raise ValueError("VM evidence must not traverse a link or junction")
    if not path.is_file() or (limit is not None and path.stat().st_size > limit):
        raise ValueError("Missing or oversized VM evidence file")
    return path


def parse_info(text):
    result = {}
    for line in text.splitlines():
        if "=" not in line:
            continue  # VirtualBox also emits a recording-screen section heading.
        key, value = line.split("=", 1)
        key = json.loads(key) if key.startswith('"') else key
        if key in result:
            raise ValueError("Duplicate VM information field")
        result[key] = json.loads(value)
    return result


def check_info(info, name, machine_uuid, config_path, disk_path):
    expected = EXPECTED | {
        "name": name, "UUID": machine_uuid, "CfgFile": str(config_path),
        "Private-SATA-0-0": str(disk_path),
    }
    expected.update({f"nic{i}": "none" for i in range(1, 37)})
    for key, value in expected.items():
        if key not in info or type(info[key]) is not type(value) or info[key] != value:
            raise ValueError(f"Unexpected or absent private VM field: {key}")
    for key in info:
        if (key.lower().startswith(("sharedfolder", "forwarding", "usbfilter"))
                or re.fullmatch(r"storagecontrollername[1-9][0-9]*", key)):
            raise ValueError("Unexpected sharing, port forwarding or storage controller")


def check_xml(text, name, machine_uuid, disk_uuid):
    root = ET.fromstring(text)
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    machines = root.findall("Machine")
    if len(machines) != 1:
        raise ValueError("Expected exactly one VM definition")
    machine = machines[0]
    if (machine.get("name") != name or machine.get("uuid") != "{" + machine_uuid + "}"
            or machine.get("OSType") != "Windows11_64"):
        raise ValueError("VM XML identity mismatch")
    forbidden = {"SharedFolder", "USBFilter", "USBController", "HostDrive",
                 "HostPci", "Snapshot", "ExtraData", "Forwarding", "BridgedInterface",
                 "HostOnlyInterface", "HostOnlyNetwork"}
    if any(element.tag in forbidden for element in root.iter()):
        raise ValueError("Unexpected host integration or alternate guest state")
    tpm = machine.find("Hardware/TrustedPlatformModule")
    if tpm is None or tpm.get("type") != "v2_0" or tpm.get("location") != "":
        raise ValueError("Expected the internal virtual TPM 2.0")
    disks = machine.findall("MediaRegistry/HardDisks/HardDisk")
    if len(disks) != 1 or disks[0].attrib != {
        "uuid": "{" + disk_uuid + "}", "location": "private-windows.vdi",
        "format": "VDI", "type": "Normal",
    }:
        raise ValueError("Unexpected disk registry")
    devices = machine.findall("Hardware/StorageControllers/StorageController/AttachedDevice")
    if len(devices) != 2:
        raise ValueError("Expected one private disk and one empty DVD drive")
    disk, dvd = devices
    if (disk.get("type") != "HardDisk" or disk.get("port") != "0"
            or disk.get("device") != "0" or disk.get("hotpluggable") != "false"
            or len(disk) != 1 or disk[0].tag != "Image"
            or disk[0].get("uuid") != "{" + disk_uuid + "}"
            or dvd.get("type") != "DVD" or dvd.get("port") != "1"
            or dvd.get("device") != "0" or dvd.get("passthrough") != "false" or len(dvd)):
        raise ValueError("Unexpected storage attachment")
    adapters = machine.findall("Hardware/Network/Adapter")
    active = [adapter for adapter in adapters if adapter.get("enabled") == "true"]
    if active:
        raise ValueError("Unexpected enabled network adapter")
    return len(devices)


def inspect(preparation, repo_root):
    preparation = ordinary_file(preparation, 128 * 1024)
    base = preparation.parent
    if (preparation.name != "preparation.json" or base.parent != repo_root / "build"
            or re.fullmatch(r"isolated-win11-[0-9a-f]{32}", base.name) is None):
        raise ValueError("Expected a project-local preparation receipt")
    plan = json.loads(preparation.read_text(encoding="utf-8-sig"))
    if (plan.get("vm_created") is not True or plan.get("guest_started") is not False
            or plan.get("manager_sha256") != MANAGER_HASH):
        raise ValueError("Invalid empty-VM preparation")
    name = "Carl-Cam-Private-Win11-" + base.name.removeprefix("isolated-win11-")[:8]
    machine_uuid = str(UUID(plan["uuid"]))
    config_path = ordinary_file(base / name / (name + ".vbox"), 256 * 1024)
    disk_path = ordinary_file(base / name / "private-windows.vdi")
    if plan["name"] != name or plan["disk"] != str(disk_path):
        raise ValueError("Preparation target mismatch")
    manager_bytes = ordinary_file(MANAGER, 32 * 1024 * 1024).read_bytes()
    if hashlib.sha256(manager_bytes).hexdigest() != MANAGER_HASH:
        raise ValueError("Manager binary changed")

    def read_command(*arguments):
        completed = subprocess.run([str(MANAGER), *arguments], capture_output=True,
                                   text=True, encoding="utf-8", timeout=20, check=True)
        return completed.stdout

    if read_command("--version").strip() != "7.2.20r175154":
        raise ValueError("Unexpected manager version")
    info_text = read_command("showvminfo", machine_uuid, "--machinereadable")
    info = parse_info(info_text)
    check_info(info, name, machine_uuid, config_path, disk_path)
    xml_bytes = config_path.read_bytes()
    disk_uuid = str(UUID(info["Private-SATA-ImageUUID-0-0"]))
    check_xml(xml_bytes, name, machine_uuid, disk_uuid)
    medium_text = read_command("showmediuminfo", "disk", str(disk_path))
    medium = dict(line.split(":", 1) for line in medium_text.splitlines() if ":" in line)
    medium = {key.strip(): value.strip() for key, value in medium.items()}
    for key, expected in {"UUID": disk_uuid, "Parent UUID": "base", "State": "created",
                          "Type": "normal (base)", "Location": str(disk_path),
                          "Storage format": "VDI", "Format variant": "dynamic default",
                          "Capacity": "65536 MBytes", "Encryption": "disabled"}.items():
        if medium.get(key) != expected:
            raise ValueError(f"Unexpected disk field: {key}")
    if disk_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Disk is no longer empty")
    if config_path.read_bytes() != xml_bytes:
        raise ValueError("Configuration changed during inspection")
    return {
        "reviewed_utc": datetime.now(UTC).isoformat(), "name": name, "uuid": machine_uuid,
        "preparation": str(preparation), "configuration_reviewed": True,
        "config_sha256": hashlib.sha256(xml_bytes).hexdigest(),
        "manager_sha256": MANAGER_HASH, "disk_bytes": disk_path.stat().st_size,
        "showvminfo": info_text, "showmediuminfo": medium_text,
        "guest_started": False, "boot_compatibility_verified": False,
        "windows_installed": False, "privacy_verified": False, "runtime_loaded": False,
        "network_test_verified": False,
        "network_adapter_enabled": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = inspect(args.preparation.absolute(), repo)
    output = args.preparation.parent / ("configuration-review-" + uuid4().hex + ".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"configuration_reviewed": True, "evidence": str(output),
                      "guest_started": False, "privacy_verified": False}))


if __name__ == "__main__":
    main()
