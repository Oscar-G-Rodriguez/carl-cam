# Build a data-only UDF optical image from the exact verified private payload.
# This uses Windows' filesystem-image API; no physical disc, mounting, guest
# action, network, shell launcher, installer or native vision SDK is invoked.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$DiscDirectory)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build'))
$discRoot = [IO.Path]::GetFullPath($DiscDirectory)
if ((Split-Path -Parent $discRoot) -cne $buildRoot -or
    (Split-Path -Leaf $discRoot) -notmatch '^vision-guest-disc-[0-9a-f]{32}$') {
    throw 'Use a generated private disc directory directly inside this repository build folder.'
}

function Assert-OrdinaryPath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked paths are forbidden.' }
        $parent = Split-Path -Parent $item.FullName
        if (-not $parent -or $parent -eq $item.FullName) { break }
        $item = Get-Item -LiteralPath $parent -Force
    }
}

$payload = Join-Path $discRoot 'payload'
$preparationPath = Join-Path $discRoot 'disc-preparation.json'
Assert-OrdinaryPath $payload
Assert-OrdinaryPath $preparationPath
$preparationHash = (Get-FileHash -LiteralPath $preparationPath -Algorithm SHA256).Hash.ToLowerInvariant()
$record = Get-Content -LiteralPath $preparationPath -Raw | ConvertFrom-Json
foreach ($key in @('payload_verified','iso_created','disc_mounted','guest_launched',
                  'vision_runtime_loaded','privacy_verified','gpu_benchmarked')) {
    if ($record.$key -isnot [bool]) { throw 'Preparation flags must be JSON booleans.' }
}
if ($record.schema_version -isnot [long] -and $record.schema_version -isnot [int]) {
    throw 'Invalid preparation schema.'
}
if ($record.schema_version -ne 1 -or $record.payload_verified -cne $true -or
    $record.iso_created -cne $false -or $record.disc_mounted -cne $false -or
    $record.guest_launched -cne $false -or $record.vision_runtime_loaded -cne $false -or
    $record.privacy_verified -cne $false -or $record.gpu_benchmarked -cne $false) {
    throw 'Expected a static, unexecuted private disc preparation.'
}

function Test-Payload {
    $known = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $record.files) {
        if (($entry.size -isnot [int] -and $entry.size -isnot [long]) -or $entry.size -lt 0 -or
            $entry.sha256 -isnot [string] -or $entry.sha256 -cnotmatch '^[0-9a-f]{64}$') {
            throw 'Invalid payload file identity.'
        }
        if ($entry.relative -isnot [string] -or $entry.relative -match '(^/|\\|:|(^|/)\.\.?(/|$)|//)' -or
            -not $entry.relative -or -not $known.Add($entry.relative)) {
            throw 'Unsafe or duplicate payload path.'
        }
        $target = [IO.Path]::GetFullPath((Join-Path $payload $entry.relative))
        if (-not $target.StartsWith($payload + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Payload path escapes its private directory.'
        }
        Assert-OrdinaryPath $target
        $item = Get-Item -LiteralPath $target -Force
        if ($item.PSIsContainer -or $item.Length -ne $entry.size -or
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.sha256) {
            throw 'Payload file identity changed.'
        }
    }
    $actualCount = 0
    foreach ($item in Get-ChildItem -LiteralPath $payload -Recurse -Force) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked payload item.' }
        if (-not $item.PSIsContainer) {
            $name = [IO.Path]::GetRelativePath($payload, $item.FullName).Replace('\','/')
            if (-not $known.Contains($name)) { throw 'Unexpected payload file.' }
            $actualCount++
        }
    }
    if ($actualCount -ne $known.Count -or
        (Get-FileHash -LiteralPath $preparationPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $preparationHash) {
        throw 'Disc preparation changed during inspection.'
    }
}

Test-Payload
$isoPath = Join-Path $discRoot 'carl-cam-vm-checks.iso'
if (Test-Path -LiteralPath $isoPath) { throw 'ISO already exists; refusing to overwrite.' }
$attemptPath = Join-Path $discRoot ('iso-attempt-' + [guid]::NewGuid().ToString('N') + '.json')
$attempt = [ordered]@{ schema_version=1; started_utc=[DateTimeOffset]::UtcNow.ToString('o');
    preparation_sha256=$preparationHash; payload_directory=$payload; iso_path=$isoPath;
    iso_created=$false; payload_rechecked=$false; disc_mounted=$false; guest_launched=$false;
    vision_runtime_loaded=$false; privacy_verified=$false; gpu_benchmarked=$false }

try {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
public static class CarlCamPrivateIsoStream {
    public static void Write(object source, string target, long expectedBytes) {
        IStream stream = (IStream)source;
        IntPtr count = Marshal.AllocHGlobal(4);
        try {
            using (FileStream output = new FileStream(target, FileMode.CreateNew, FileAccess.Write)) {
                byte[] buffer = new byte[1024 * 1024];
                long written = 0;
                while (written < expectedBytes) {
                    int request = (int)Math.Min(buffer.Length, expectedBytes - written);
                    Marshal.WriteInt32(count, 0);
                    stream.Read(buffer, request, count);
                    int received = Marshal.ReadInt32(count);
                    if (received <= 0 || received > request) throw new IOException("Truncated ISO stream");
                    output.Write(buffer, 0, received);
                    written += received;
                }
                output.Flush(true);
            }
        } finally { Marshal.FreeHGlobal(count); }
    }
}
'@
    $image = New-Object -ComObject IMAPI2FS.MsftFileSystemImage
    $image.FileSystemsToCreate = 4 # UDF supports the original long Python paths.
    $image.VolumeName = 'CARLCAMCHECKS'
    $image.FreeMediaBlocks = 400000
    $image.Root.AddTree($payload, $false)
    $result = $image.CreateResultImage()
    $expectedBytes = [long]$result.BlockSize * [long]$result.TotalBlocks
    if ($expectedBytes -le 0 -or $expectedBytes -gt 800MB) { throw 'Unexpected optical-image size.' }
    [CarlCamPrivateIsoStream]::Write($result.ImageStream, $isoPath, $expectedBytes)
    if ((Get-Item -LiteralPath $isoPath).Length -ne $expectedBytes) { throw 'ISO length mismatch.' }
    Test-Payload
    $attempt.payload_rechecked = $true
    $attempt.iso_sha256 = (Get-FileHash -LiteralPath $isoPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $attempt.iso_bytes = $expectedBytes
    $attempt.iso_created = $true
    $attempt.file_system = 'UDF'
    $attempt.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
    $attempt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $attemptPath -Encoding utf8
    Write-Output ('Private read-only data ISO: ' + $isoPath)
    Write-Output ('SHA-256: ' + $attempt.iso_sha256)
    Write-Output ('Receipt: ' + $attemptPath)
} catch {
    $attempt.error = $_.Exception.Message
    $attempt.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
    $attempt.partial_iso_retained = Test-Path -LiteralPath $isoPath
    $attempt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $attemptPath -Encoding utf8
    throw
}
