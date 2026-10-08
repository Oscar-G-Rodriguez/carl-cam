"""Read-only inspection of the three-CPU private guest and its selected DVD.

This does not establish a working Windows installation, a license, or native
runtime privacy. It never starts a guest, attaches media, or imports a SDK.
"""

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from inspect_windows_test_vm import EXPECTED, MANAGER, MANAGER_HASH, ordinary_file

NAME = "Carl-Cam-Private-Win11-bee3f3e4"
UUID = "2cb2eaf1-068d-47a3-8539-8b0d9084b32e"
DISK_UUID = "3907b0bd-2f99-405c-a6b8-534142c8a3fe"
VM_DIRECTORY = Path("build/isolated-win11-bee3f3e413594545b4ce85b76e3f333c") / NAME
CHECKS_DIRECTORY = Path("build/vision-guest-disc-f0c58591536e4e00aa2ec220dad46dbd")
CHECKS_HASH = "da50e04dfbe04d415e462cbb806c5c2eed342176f01986e52cce6bb05cca12e0"
CHECKS_DVD_UUID = "41c96984-d18e-4cb4-b347-07300aa8e525"
OBSERVER_DIRECTORY = Path("build/vision-guest-disc-d9ef8052e9ce4e619954da1e36c81230")
OBSERVER_HASH = "714f442a28b506922a875503b786d293136ee49562e98ff493b6d2d7315a615c"
WINDOWS_HASH = "bc3f24086ebadc94489066b5ad78089e2cf5c3491e90e790bb81a2b199c10e38"
WINDOWS_DVD_UUID = "0f6013bc-a7c5-46b1-ae2d-5eddefc20e17"
WINDOWS_ISO = Path.home() / "Downloads" / (
    "26300.9457.260913-1737.26h2_ge_release_svc_refresh_CLIENTENTERPRISEEVAL_OEMRET_x64FRE_en-us.iso"
)


def file_hash(path):
    with ordinary_file(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_information(text):
    info = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = json.loads(key) if key.startswith('"') else key
        if key in info:
            raise ValueError("Duplicate VM information field")
        info[key] = value if key in {"VideoMode", "VRDEClients"} else json.loads(value)
    return info


def check_information(info, repo, dvd, state):
    if state not in {"poweroff", "running"}:
        raise ValueError("Use an explicit powered-off or running inspection")
    expected = EXPECTED | {
        "name": NAME, "UUID": UUID, "hardwareuuid": UUID, "VMState": state, "cpus": 3,
        "CfgFile": str(repo / VM_DIRECTORY / (NAME + ".vbox")),
        "Private-SATA-0-0": str(repo / VM_DIRECTORY / "private-windows.vdi"),
        "Private-SATA-ImageUUID-0-0": DISK_UUID, "Private-SATA-1-0": str(dvd),
        "Private-SATA-IsEjected-1-0": "off", "Private-SATA-tempeject-1-0": "off",
        "storagecontrollerinstance0": "0",
        **{f"nic{i}": "none" for i in range(1, 37)},
    }
    for key, value in expected.items():
        if key not in info or type(info[key]) is not type(value) or info[key] != value:
            raise ValueError(f"Unexpected or absent private guest field: {key}")
    for key in info:
        if (key.lower().startswith(("sharedfolder", "forwarding", "usbfilter", "pcidevice"))
                or re.fullmatch(r"storagecontrollername[1-9][0-9]*", key)
                or re.fullmatch(r"Private-SATA-(?![01]-0$)[0-9]+-[0-9]+", key)):
            raise ValueError("Unexpected host integration or storage attachment")
    dvd_uuid = info.get("Private-SATA-ImageUUID-1-0")
    if not isinstance(dvd_uuid, str) or re.fullmatch(
        r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", dvd_uuid
    ) is None:
        raise ValueError("Invalid DVD identity")
    return dvd_uuid


def check_configuration(data, dvd, dvd_uuid):
    root = ET.fromstring(data)
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    machines = root.findall("Machine")
    if len(machines) != 1:
        raise ValueError("Expected exactly one private guest")
    machine = machines[0]
    if (machine.get("name") != NAME or machine.get("uuid") != "{" + UUID + "}"
            or machine.get("hardwareUUID", machine.get("uuid")) != "{" + UUID + "}"
            or machine.get("OSType") != "Windows11_64"):
        raise ValueError("Configuration identity mismatch")
    forbidden = {"SharedFolder", "USBFilter", "USBController", "HostDrive", "HostPci",
                 "Snapshot", "Forwarding", "BridgedInterface", "HostOnlyInterface",
                 "HostOnlyNetwork"}
    if any(element.tag in forbidden for element in root.iter()):
        raise ValueError("Unexpected host integration or alternate guest state")
    # The GUI writes its window position after use. Permit only that display
    # metadata; unknown extra-data settings must still be reviewed separately.
    for extra in root.iter("ExtraData"):
        for item in extra:
            if (item.tag != "ExtraDataItem" or set(item.attrib) != {"name", "value"}
                    or item.get("name") != "GUI/LastNormalWindowPosition"
                    or re.fullmatch(r"-?\d+,-?\d+,\d+,\d+(?:,max)?", item.get("value", "")) is None
                    or len(item)):
                raise ValueError("Unreviewed extra-data setting")
    for path, required in {
        "Hardware/TrustedPlatformModule": {"type": "v2_0", "location": ""},
        "Hardware/CPU": {"count": "3"}, "Hardware/Memory": {"RAMSize": "4096"},
        "Hardware/Firmware": {"type": "EFI"}, "Hardware/Chipset": {"type": "ICH9"},
    }.items():
        elements = machine.findall(path)
        if len(elements) != 1 or any(elements[0].get(k) != v for k, v in required.items()):
            raise ValueError(f"Unexpected private guest hardware: {path}")
    if any(a.get("enabled") not in {None, "false"}
           for a in machine.findall("Hardware/Network/Adapter")):
        raise ValueError("Enabled or unrecognized network adapter")
    disks = machine.findall("MediaRegistry/HardDisks/HardDisk")
    if len(disks) != 1 or disks[0].attrib != {
        "uuid": "{" + DISK_UUID + "}", "location": "private-windows.vdi",
        "format": "VDI", "type": "Normal",
    } or len(disks[0]):
        raise ValueError("Unexpected disk registry")
    images = machine.findall("MediaRegistry/DVDImages/Image")
    observer_selected = tuple(dvd.parts[-3:]) == (*OBSERVER_DIRECTORY.parts, "carl-cam-vm-checks.iso")
    if not 1 <= len(images) <= (3 if observer_selected else 2):
        raise ValueError("Unexpected DVD registry")
    selected = 0
    known = set()
    for image in images:
        identity = image.get("uuid")
        location = Path(image.get("location", ""))
        if set(image.attrib) != {"uuid", "location"} or len(image) or identity in known:
            raise ValueError("Unexpected DVD registry")
        known.add(identity)
        if identity == "{" + dvd_uuid + "}" and location == dvd:
            selected += 1
        elif (observer_selected and location == dvd.parents[2] / CHECKS_DIRECTORY / "carl-cam-vm-checks.iso"
              and identity == "{" + CHECKS_DVD_UUID + "}"):
            pass
        elif not (dvd.name == "carl-cam-vm-checks.iso" and location == WINDOWS_ISO
                  and identity == "{" + WINDOWS_DVD_UUID + "}"):
            raise ValueError("Unexpected inactive DVD registry entry")
    if selected != 1:
        raise ValueError("Expected exactly one selected DVD registry entry")
    controllers = machine.findall("Hardware/StorageControllers/StorageController")
    if (len(controllers) != 1 or controllers[0].get("name") != "Private-SATA"
            or controllers[0].get("type") != "AHCI" or controllers[0].get("PortCount") != "2"):
        raise ValueError("Unexpected storage controller")
    devices = list(controllers[0])
    if len(devices) != 2:
        raise ValueError("Expected only the private disk and selected DVD")
    for device, kind, port, identity in zip(
        devices, ("HardDisk", "DVD"), ("0", "1"), (DISK_UUID, dvd_uuid), strict=True
    ):
        if (device.tag != "AttachedDevice" or device.get("type") != kind
                or device.get("port") != port or device.get("device") != "0"
                or device.get("hotpluggable") != "false"
                or (kind == "DVD" and device.get("passthrough") != "false")
                or len(device) != 1 or device[0].tag != "Image"
                or device[0].attrib != {"uuid": "{" + identity + "}"}):
            raise ValueError("Unexpected storage attachment")


def inspect(repo, disc, state):
    if disc not in {"installation", "checks", "observer"}:
        raise ValueError("Select a reviewed DVD")
    dvd, checksum = {
        "installation": (WINDOWS_ISO, WINDOWS_HASH),
        "checks": (repo / CHECKS_DIRECTORY / "carl-cam-vm-checks.iso", CHECKS_HASH),
        "observer": (repo / OBSERVER_DIRECTORY / "carl-cam-vm-checks.iso", OBSERVER_HASH),
    }[disc]
    if file_hash(MANAGER) != MANAGER_HASH or file_hash(dvd) != checksum:
        raise ValueError("Manager or selected DVD bytes changed")

    def read(*arguments):
        return subprocess.run([str(MANAGER), *arguments], capture_output=True, text=True,
                              encoding="utf-8", timeout=20, check=True).stdout

    if read("--version").strip() != "7.2.20r175154":
        raise ValueError("Manager version changed")
    info = parse_information(read("showvminfo", UUID, "--machinereadable"))
    dvd_uuid = check_information(info, repo, dvd, state)
    config = ordinary_file(repo / VM_DIRECTORY / (NAME + ".vbox"), 256 * 1024)
    ordinary_file(repo / VM_DIRECTORY / "private-windows.vdi")
    data = config.read_bytes()
    check_configuration(data, dvd, dvd_uuid)
    after = parse_information(read("showvminfo", UUID, "--machinereadable"))
    check_information(after, repo, dvd, state)
    if (after.get("Private-SATA-ImageUUID-1-0") != dvd_uuid
            or config.read_bytes() != data or file_hash(dvd) != checksum):
        raise ValueError("Guest configuration or DVD changed during inspection")
    return {"configuration_reviewed": True, "name": NAME, "uuid": UUID,
            "observed_state": state, "selected_disc": disc, "dvd_sha256": checksum,
            "configuration_sha256": hashlib.sha256(data).hexdigest(),
            "vm_started_by_check": False, "media_attached_by_check": False,
            "guest_files_verified": False, "windows_license_verified": False,
            "vision_runtime_loaded": False, "privacy_verified": False,
            "gpu_benchmarked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disc", choices=("installation", "checks", "observer"), required=True)
    parser.add_argument("--state", choices=("poweroff", "running"), required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(Path(__file__).resolve().parents[1], args.disc, args.state), indent=2))


if __name__ == "__main__":
    main()
