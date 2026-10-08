# Keep the vision compiler local to Carl-Cam; change no permanent environment settings.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$toolDirectory = Join-Path $repoRoot 'build\tools\bazel'
New-Item -ItemType Directory -Path $toolDirectory -Force | Out-Null
$releaseUrl = 'https://releases.bazel.build/7.4.1/release/bazel-7.4.1-windows-x86_64.exe'
$checksumPath = Join-Path $toolDirectory 'bazel-7.4.1-windows-x86_64.exe.sha256'
$binaryPath = Join-Path $toolDirectory 'bazel-7.4.1-windows-x86_64.exe'
Invoke-WebRequest -Uri ($releaseUrl + '.sha256') -OutFile $checksumPath -UseBasicParsing
$publishedChecksum = (Get-Content -LiteralPath $checksumPath -Raw).Trim()
if ($publishedChecksum -notmatch '^([0-9a-fA-F]{64})\s+\*?bazel-7\.4\.1-windows-x86_64\.exe$') {
    throw 'Official release checksum has an unexpected format.'
}
$expectedHash = $Matches[1].ToLowerInvariant()
if (Test-Path -LiteralPath $binaryPath) {
    $actualHash = (Get-FileHash -LiteralPath $binaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw 'Existing Bazel binary differs from the official checksum; preserve it for inspection.'
    }
} else {
    Invoke-WebRequest -Uri $releaseUrl -OutFile $binaryPath -UseBasicParsing
    $actualHash = (Get-FileHash -LiteralPath $binaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) { throw 'Downloaded Bazel binary failed checksum verification.' }
}
$versionText = & $binaryPath --version 2>&1
$versionCode = $LASTEXITCODE
if ($versionCode -ne 0 -or ($versionText -join "`n") -ne 'bazel 7.4.1') {
    throw 'Verified Bazel binary failed the exact version check.'
}
$record = [ordered]@{
    verifiedAt = [DateTimeOffset]::Now.ToString('o')
    releaseUrl = $releaseUrl
    checksumUrl = $releaseUrl + '.sha256'
    publishedChecksum = $publishedChecksum
    binaryPath = $binaryPath
    sha256 = $actualHash
    bytes = (Get-Item -LiteralPath $binaryPath).Length
    versionOutput = $versionText -join "`n"
    versionExitCode = $versionCode
    verification = 'Exact SHA-256 matched the publisher HTTPS release checksum; no global PATH or registry changes.'
}
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $toolDirectory 'provenance.json') -Encoding utf8
$record | ConvertTo-Json -Depth 4
