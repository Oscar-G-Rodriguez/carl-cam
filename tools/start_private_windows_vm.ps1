# Attach the publisher-verified evaluation ISO and boot only the existing private
# guest. No Windows installation, networking, host integration or policy change.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$IsoPath,
    [switch]$Start,
    [switch]$CheckRunning,
    [ValidateSet(2,3)][int]$ExpectedCpuCount = 2
)

$ErrorActionPreference = 'Stop'
if ($Start -and $CheckRunning) { throw 'Choose startup or read-only running inspection, not both.' }
if ($Start -and $ExpectedCpuCount -ne 2) {
    throw 'Initial empty-guest startup is restricted to the original two-CPU preparation. Use three CPUs only for an explicitly approved running inspection.'
}
$repoRoot = Split-Path -Parent $PSScriptRoot
$manager = 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe'
$expectedManagerHash = '6852390a57d454d3b7c9ae27e66e45c367699faf2807b450de084b1f7d0203c1'
$expectedIsoHash = 'bc3f24086ebadc94489066b5ad78089e2cf5c3491e90e790bb81a2b199c10e38'
$machineUuid = '2cb2eaf1-068d-47a3-8539-8b0d9084b32e'
$preparation = Join-Path $repoRoot 'build\isolated-win11-bee3f3e413594545b4ce85b76e3f333c\preparation.json'
$expectedDisk = Join-Path (Split-Path -Parent $preparation) 'Carl-Cam-Private-Win11-bee3f3e4\private-windows.vdi'
$expectedName = 'Carl-Cam-Private-Win11-bee3f3e4'
$expectedConfiguration = Join-Path (Split-Path -Parent $expectedDisk) ($expectedName + '.vbox')
$expectedSigner = 'CN="Oracle America, Inc.", O="Oracle America, Inc.", L=Redwood City, S=California, C=US'
$signature = Get-AuthenticodeSignature -LiteralPath $manager
if ((Get-FileHash -LiteralPath $manager).Hash.ToLowerInvariant() -ne $expectedManagerHash -or
    $signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cne $expectedSigner) {
    throw 'The VirtualBox manager identity is not the verified installation.'
}
$isoItem = Get-Item -LiteralPath $IsoPath
if ($isoItem.PSIsContainer -or ($isoItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'The installation ISO must be an ordinary file.'
}
$verifiedIsoPath = $isoItem.FullName
if ((Get-FileHash -LiteralPath $verifiedIsoPath).Hash.ToLowerInvariant() -ne $expectedIsoHash) {
    throw 'The Windows ISO does not match the published English-US x64 evaluation hash.'
}
$hostMemory = Get-CimInstance Win32_OperatingSystem
$freeBytes = (Get-PSDrive C).Free
if ($freeBytes -lt 45GB) { throw 'Insufficient disk headroom for the private Windows guest.' }
if (-not $Start -and -not $CheckRunning) {
    [ordered]@{ vm_uuid=$machineUuid; iso=$verifiedIsoPath; iso_sha256=$expectedIsoHash;
        available_memory_kib=$hostMemory.FreePhysicalMemory; free_disk_bytes=$freeBytes;
        low_memory_override_requested=$true; expected_cpu_count=$ExpectedCpuCount;
        guest_started=$false } | ConvertTo-Json
    return
}

# The owner explicitly requested proceeding with current memory headroom. Do
# not close host apps, alter page-file settings, or reduce the approved guest RAM.
$outputDirectory = Join-Path $repoRoot ('build\windows-boot-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $outputDirectory -ErrorAction Stop | Out-Null
$receipt = [ordered]@{ requested_utc=[DateTimeOffset]::UtcNow.ToString('o');
    operation=$(if ($Start) { 'attach-and-start' } else { 'read-only-running-inspection' });
    vm_uuid=$machineUuid; iso=$verifiedIsoPath; iso_sha256=$expectedIsoHash;
    expected_cpu_count=$ExpectedCpuCount;
    iso_bytes=$isoItem.Length; manager_sha256=$expectedManagerHash;
    available_memory_kib=$hostMemory.FreePhysicalMemory; free_disk_bytes=$freeBytes;
    low_memory_override_requested=$true; commands=@(); iso_attached=$false;
    guest_start_command_succeeded=$false; guest_running=$false; windows_installed=$false;
    runtime_loaded=$false; privacy_verified=$false; gpu_benchmarked=$false }

function Invoke-PrivateManager([string[]]$CommandArguments) {
    $result = & $manager @CommandArguments 2>&1
    $resultCode = $LASTEXITCODE
    $receipt.commands += [ordered]@{ arguments=$CommandArguments; exit_code=$resultCode;
        output=($result -join "`n"); utc=[DateTimeOffset]::UtcNow.ToString('o') }
    if ($resultCode -ne 0) { throw ('VirtualBox command failed: ' + ($result -join "`n")) }
    return ($result -join "`n")
}

function Read-PrivateConfiguration {
    $text = Invoke-PrivateManager @('showvminfo',$machineUuid,'--machinereadable')
    $info = @{}
    foreach ($line in ($text -split "`n")) {
        $line = $line.TrimEnd("`r")
        if ($line -notmatch '=') { continue }
        $parts = $line -split '=',2
        $key = if ($parts[0].StartsWith('"')) { $parts[0] | ConvertFrom-Json } else { $parts[0] }
        if ($info.ContainsKey($key)) { throw 'Duplicate VM configuration key.' }
        # Running VirtualBox 7.2 guests append display/client fields whose
        # values are not JSON (e.g. VideoMode="0,0,0"@0,0 1; VRDEClients==0).
        # These two diagnostics are not isolation policy; retain their raw text.
        if ($key -in @('VideoMode','VRDEClients')) { $info[$key] = $parts[1] }
        else { $info[$key] = $parts[1] | ConvertFrom-Json }
    }
    foreach ($number in 1..36) { if ($info["nic$number"] -ne 'none') { throw 'Guest network is enabled.' } }
    foreach ($key in @('clipboard','draganddrop')) {
        if ($info[$key] -ne 'disabled') { throw 'Guest integration is enabled.' }
    }
    foreach ($key in @('usb','ehci','xhci','audio_in','audio_out','vrde','recording_enabled',
        'recording_started','clipboard_file_transfers','autostart-enabled','tracing-enabled',
        'tracing-allow-vm-access')) {
        if ($info[$key] -ne 'off') { throw 'Unexpected guest integration.' }
    }
    # Do not let PowerShell coerce a JSON string into a valid resource count.
    if (($info.memory -isnot [int] -and $info.memory -isnot [long]) -or
        ($info.cpus -isnot [int] -and $info.cpus -isnot [long]) -or
        $info.memory -ne 4096 -or $info.cpus -ne $ExpectedCpuCount -or
        $info.UUID -cne $machineUuid -or $info.name -cne $expectedName -or
        $info.CfgFile -cne $expectedConfiguration -or
        $info.audio -ne 'none' -or $info.SecureBoot -ne 'on' -or
        $info.storagecontrollername0 -ne 'Private-SATA' -or
        $info['Private-SATA-0-0'] -ne $expectedDisk -or
        @($info.Keys | Where-Object {
            $_ -match '^(sharedfolder|forwarding|usbfilter|storagecontrollername[1-9])'
        }).Count) {
        throw 'Unexpected private guest identity, resources or sharing.'
    }
    return $info
}

try {
    if ($Start) {
      $auditOutput = & (Join-Path $repoRoot '.venv\Scripts\python.exe') `
        (Join-Path $repoRoot 'tools\inspect_windows_test_vm.py') --preparation $preparation 2>&1
    if ($LASTEXITCODE -ne 0) { throw ('Preboot configuration audit failed: ' + ($auditOutput -join "`n")) }
    $receipt.preboot_audit = $auditOutput -join "`n"
    $before = Read-PrivateConfiguration
    if ($before.VMState -ne 'poweroff' -or $before['Private-SATA-1-0'] -ne 'emptydrive') {
        throw 'Expected the existing powered-off empty guest with no ISO attached.'
    }
    Copy-Item -LiteralPath $before.CfgFile -Destination (Join-Path $outputDirectory 'before.vbox')
    Invoke-PrivateManager @('storageattach',$machineUuid,'--storagectl','Private-SATA',
        '--port','1','--device','0','--type','dvddrive','--medium',$verifiedIsoPath,
        '--passthrough','off') | Out-Null
    $attached = Read-PrivateConfiguration
    if ($attached['Private-SATA-1-0'] -ne $verifiedIsoPath -or $attached.VMState -ne 'poweroff') {
        throw 'Windows ISO attachment did not read back correctly.'
    }
    $receipt.iso_attached = $true
    $receipt.attached_config_sha256 = (Get-FileHash -LiteralPath $attached.CfgFile).Hash.ToLowerInvariant()
      Invoke-PrivateManager @('startvm',$machineUuid,'--type','gui') | Out-Null
      $receipt.guest_start_command_succeeded = $true
    }
    $started = Read-PrivateConfiguration
    if ($started['Private-SATA-1-0'] -ne $verifiedIsoPath) { throw 'Unexpected running guest ISO.' }
    $receipt.iso_attached = $true
    $receipt.guest_running = $started.VMState -eq 'running'
    $receipt.vm_state = $started.VMState
    $receipt.configuration_sha256 = (Get-FileHash -LiteralPath $expectedConfiguration).Hash.ToLowerInvariant()
    $receipt.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
    if (-not $receipt.guest_running) { throw 'The guest did not reach a running VM state.' }
    Write-Output 'Verified Windows ISO attached; private VM is running with host integrations disabled.'
} catch {
    $receipt.error = $_.Exception.Message
    throw
} finally {
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputDirectory 'attempt.json') -Encoding utf8
    Write-Output ('Private Windows boot evidence: ' + $outputDirectory)
}
