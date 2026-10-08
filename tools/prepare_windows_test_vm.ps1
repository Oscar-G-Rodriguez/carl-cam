# Create only a new, powered-off private test machine. No ISO download, OS
# installation, guest launch, shared folder, camera, trace or host-policy change.
[CmdletBinding()]
param([switch]$Prepare)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$manager = 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe'
$expectedManagerHash = '6852390a57d454d3b7c9ae27e66e45c367699faf2807b450de084b1f7d0203c1'
$expectedSigner = 'CN="Oracle America, Inc.", O="Oracle America, Inc.", L=Redwood City, S=California, C=US'
if (-not (Test-Path -LiteralPath $manager)) { throw 'The verified VirtualBox installation is required.' }
$managerIdentity = (Get-FileHash -LiteralPath $manager -Algorithm SHA256).Hash.ToLowerInvariant()
$signature = Get-AuthenticodeSignature -LiteralPath $manager
if ($managerIdentity -ne $expectedManagerHash -or $signature.Status -ne 'Valid' -or
    $signature.SignerCertificate.Subject -cne $expectedSigner) {
    throw 'The manager identity changed; inspect it before preparing a guest.'
}
$version = & $manager --version
if ($LASTEXITCODE -ne 0 -or $version -ne '7.2.20r175154') { throw 'Unexpected manager version.' }
$existingMachines = & $manager list vms
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect registered machines.' }
if (($existingMachines -join "`n") -match 'Carl-Cam-Private-Win11-') {
    throw 'A private test VM already exists. Inspect its receipt instead of accumulating machines.'
}
$freeBytes = (Get-PSDrive -Name C).Free
if ($freeBytes -lt 45GB) { throw 'Insufficient C: headroom; do not delete evidence or owner files.' }
$preparationId = [guid]::NewGuid().ToString('N')
$vmName = 'Carl-Cam-Private-Win11-' + $preparationId.Substring(0, 8)
$vmUuid = [guid]::NewGuid().ToString()
$baseDirectory = Join-Path $repoRoot ('build\isolated-win11-' + $preparationId)
$vmDirectory = Join-Path $baseDirectory $vmName
$diskPath = Join-Path $vmDirectory 'private-windows.vdi'
$plan = [ordered]@{
    requested_utc = [DateTimeOffset]::UtcNow.ToString('o')
    name = $vmName
    uuid = $vmUuid
    base_directory = $baseDirectory
    manager_sha256 = $managerIdentity
    manager_version = $version
    host_free_bytes = $freeBytes
    guest_memory_mib = 4096
    guest_cpus = 2
    dynamic_disk_capacity_mib = 65536
    disk = $diskPath
    vm_created = $false
    configuration_reviewed = $false
    guest_started = $false
    windows_installed = $false
    runtime_loaded = $false
    privacy_verified = $false
    network_adapter_enabled = $false
    commands = @()
}
if (-not $Prepare) { $plan | ConvertTo-Json -Depth 8; return }

# The output is exclusively a fresh project-local child, never an existing VM.
if (Test-Path -LiteralPath $baseDirectory) { throw 'The output directory must be fresh.' }
$buildDirectory = Join-Path $repoRoot 'build'
$buildItem = Get-Item -LiteralPath $buildDirectory
if (-not $buildItem.PSIsContainer -or
    ($buildItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'The project build root must be an ordinary directory.'
}
New-Item -ItemType Directory -Path $baseDirectory | Out-Null
$receiptPath = Join-Path $baseDirectory 'preparation.json'
function Save-Receipt {
    $plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
}
function Invoke-Manager([string[]]$CommandArguments) {
    $output = & $manager @CommandArguments 2>&1
    $commandExit = $LASTEXITCODE
    $plan.commands += [ordered]@{
        arguments = $CommandArguments
        exit_code = $commandExit
        output = $output -join "`n"
    }
    Save-Receipt
    if ($commandExit -ne 0) { throw ('VM preparation command failed: ' + $CommandArguments[0]) }
}
Save-Receipt
try {
    Invoke-Manager @('createvm', '--name', $vmName, '--uuid', $vmUuid,
        '--platform-architecture', 'x86', '--ostype', 'Windows11_64',
        '--basefolder', $baseDirectory, '--register')
    $plan.vm_created = $true
    Save-Receipt
    Invoke-Manager @('modifyvm', $vmUuid, '--memory', '4096', '--cpus', '2',
        '--firmware', 'efi', '--chipset', 'ich9', '--ioapic', 'on', '--tpm-type', '2.0',
        '--graphicscontroller', 'vboxsvga', '--vram', '128', '--accelerate-3d', 'off',
        '--boot1', 'dvd', '--boot2', 'disk', '--boot3', 'none', '--boot4', 'none',
        '--mouse', 'ps2', '--keyboard', 'ps2', '--usb-ohci', 'off', '--usb-ehci', 'off',
        '--usb-xhci', 'off', '--audio-enabled', 'off', '--audio-in', 'off', '--audio-out', 'off',
        '--clipboard-mode', 'disabled', '--clipboard-file-transfers', 'disabled',
        '--drag-and-drop', 'disabled', '--vrde', 'off', '--recording', 'off',
        '--autostart-enabled', 'off', '--nic1', 'none', '--cable-connected1', 'off',
        '--nic-promisc1', 'deny', '--nic-trace1', 'off',
        '--nic2', 'none', '--nic3', 'none', '--nic4', 'none', '--nic5', 'none',
        '--nic6', 'none', '--nic7', 'none', '--nic8', 'none')
    # These initialize only the new guest's blank firmware store. They neither
    # reset an existing guest nor modify Secure Boot on the owner's computer.
    Invoke-Manager @('modifynvram', $vmUuid, 'inituefivarstore')
    Invoke-Manager @('modifynvram', $vmUuid, 'enrollmssignatures')
    Invoke-Manager @('modifynvram', $vmUuid, 'enrollorclpk')
    Invoke-Manager @('modifynvram', $vmUuid, 'secureboot', '--enable')
    Invoke-Manager @('createmedium', 'disk', '--filename', $diskPath, '--size', '65536',
        '--format', 'VDI', '--variant', 'Standard')
    Invoke-Manager @('storagectl', $vmUuid, '--name', 'Private-SATA', '--add', 'sata',
        '--controller', 'IntelAHCI', '--portcount', '2', '--hostiocache', 'off')
    Invoke-Manager @('storageattach', $vmUuid, '--storagectl', 'Private-SATA', '--port', '0',
        '--device', '0', '--type', 'hdd', '--medium', $diskPath, '--nonrotational', 'on')
    Invoke-Manager @('storageattach', $vmUuid, '--storagectl', 'Private-SATA', '--port', '1',
        '--device', '0', '--type', 'dvddrive', '--medium', 'emptydrive')
    Invoke-Manager @('showvminfo', $vmUuid, '--machinereadable')
    # showvminfo reports the configured Secure Boot policy. SecureBoot's runtime
    # UEFI variable may not exist before first boot; do not claim boot verification.
    $plan.commands_completed = $true
    Save-Receipt
    $python = Join-Path $repoRoot '.venv\Scripts\python.exe'
    $inspector = Join-Path $PSScriptRoot 'inspect_windows_test_vm.py'
    $reviewOutput = & $python $inspector --preparation $receiptPath
    if ($LASTEXITCODE -ne 0) { throw 'The independent powered-off configuration inspection failed.' }
    $review = $reviewOutput | ConvertFrom-Json
    if ($review.configuration_reviewed -ne $true -or $review.guest_started -ne $false -or
        $review.privacy_verified -ne $false) { throw 'Unexpected configuration inspection result.' }
    $plan.configuration_reviewed = $true
    $plan.configuration_review = $review.evidence
} catch {
    $plan.error = $_.Exception.Message
    throw
} finally {
    $plan.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
    Save-Receipt
    $plan | ConvertTo-Json -Depth 8
}
