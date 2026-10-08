# Download and verify the approved VM host installer; installation requires the
# explicit -Install switch and Windows' ordinary administrator approval.
# Never start a guest, install optional networking/USB/Python features or reboot.
[CmdletBinding()]
param([switch]$Install)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$toolDirectory = Join-Path $repoRoot 'build\tools\virtualbox-7.2.20'
$installerName = 'VirtualBox-7.2.20-175154-Win.exe'
$releaseBase = 'https://download.virtualbox.org/virtualbox/7.2.20/'
$expectedHash = 'a81777d2b36380ce042a29e9c554cf032eb46a793f62e3cc82e7411e535c2c26'
$expectedSigner = 'CN="Oracle America, Inc.", O="Oracle America, Inc.", L=Redwood City, S=California, C=US'
$expectedCertificate = '2B1BD188256EB612578949A943DE0DDC52D1DF79'
$installerPath = Join-Path $toolDirectory $installerName
if (Test-Path -LiteralPath $toolDirectory) {
    $existingDirectory = Get-Item -LiteralPath $toolDirectory
    if (-not $existingDirectory.PSIsContainer -or
        ($existingDirectory.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Installer preparation must use an ordinary project-local directory.'
    }
} else {
    New-Item -ItemType Directory -Path $toolDirectory | Out-Null
}
$checksumResponse = Invoke-WebRequest -Uri ($releaseBase + 'SHA256SUMS') -UseBasicParsing
$lines = @($checksumResponse.Content -split "`n" | Where-Object {
    $_.Trim() -match ('^[0-9a-f]{64}\s+\*?' + [regex]::Escape($installerName) + '$')
})
if ($lines.Count -ne 1 -or $lines[0].Trim().Split(' ')[0] -ne $expectedHash) {
    throw 'The publisher checksum does not match the reviewed installer identity.'
}
if (-not (Test-Path -LiteralPath $installerPath)) {
    Invoke-WebRequest -Uri ($releaseBase + $installerName) -OutFile $installerPath -UseBasicParsing
}
$installerFile = Get-Item -LiteralPath $installerPath
if ($installerFile.PSIsContainer -or ($installerFile.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
    $installerFile.Length -le 0 -or $installerFile.Length -gt 512MB) {
    throw 'Unexpected installer file. Preserve it for inspection.'
}
$actualHash = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) { throw 'The installer failed publisher checksum verification.' }
$signature = Get-AuthenticodeSignature -LiteralPath $installerPath
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cne $expectedSigner -or
    $signature.SignerCertificate.Thumbprint -cne $expectedCertificate) {
    throw 'A valid Oracle publisher signature is required before executing this installer.'
}
$record = [ordered]@{
    verified_utc = [DateTimeOffset]::UtcNow.ToString('o')
    installer = $installerPath
    bytes = $installerFile.Length
    sha256 = $actualHash
    release_url = $releaseBase + $installerName
    checksum_url = $releaseBase + 'SHA256SUMS'
    publisher_checksum_line = $lines[0].Trim()
    signature_status = [string]$signature.Status
    signer_subject = $signature.SignerCertificate.Subject
    signer_thumbprint = $signature.SignerCertificate.Thumbprint
    installer_executed = $false
    host_settings_changed = $false
    vm_started = $false
    restart_allowed = $false
}
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $toolDirectory 'download-provenance.json') -Encoding utf8
$record | ConvertTo-Json -Depth 4

if ($Install) {
    $installedManager = 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe'
    if (Test-Path -LiteralPath $installedManager) {
        throw 'VirtualBox already exists. Inspect it before requesting another installation.'
    }
    $attemptDirectory = Join-Path $toolDirectory ('installation-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $attemptDirectory | Out-Null
    $logPath = Join-Path $attemptDirectory 'installer.log'
    $properties = 'ADDLOCAL=VBoxApplication REBOOT=ReallySuppress VBOX_START=0 VBOX_INSTALLDESKTOPSHORTCUT=0 VBOX_INSTALLQUICKLAUNCHSHORTCUT=0 VBOX_REGISTERFILEEXTENSIONS=0'
    $argumentString = '--silent --no-silent-cert --logging --msi-log-file "' + $logPath + '" --msiparams "' + $properties + '"'
    $attempt = [ordered]@{
        requested_utc = [DateTimeOffset]::UtcNow.ToString('o')
        installer_sha256 = $actualHash
        arguments = $argumentString
        planned_features = @('VBoxApplication')
        approval_required = $true
        installer_started = $false
        installation_verified = $false
        vm_started = $false
        restart_allowed = $false
        network_extras_requested = $false
        usb_feature_requested = $false
        automatic_certificate_install_requested = $false
    }
    $attemptPath = Join-Path $attemptDirectory 'attempt.json'
    $attempt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $attemptPath -Encoding utf8
    Write-Output 'Starting the signed core-only installer. Windows administrator approval is required; no automatic restart is allowed.'
    try {
        $installerProcess = Start-Process -FilePath $installerPath -ArgumentList $argumentString -Verb RunAs -WindowStyle Hidden -PassThru
        $attempt.installer_started = $true
        $attempt.process_id = $installerProcess.Id
        $attempt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $attemptPath -Encoding utf8
        $installerProcess.WaitForExit()
        $attempt.exit_code = $installerProcess.ExitCode
        $attempt.restart_required = $installerProcess.ExitCode -eq 3010
        if ($installerProcess.ExitCode -notin @(0, 3010)) {
            throw 'The VirtualBox installer did not report successful completion. Inspect the retained log.'
        }
        if (-not (Test-Path -LiteralPath $installedManager)) {
            throw 'The installer returned without the expected manager binary.'
        }
        $managerSignature = Get-AuthenticodeSignature -LiteralPath $installedManager
        if ($managerSignature.Status -ne 'Valid' -or $managerSignature.SignerCertificate.Subject -cne $expectedSigner) {
            throw 'Installed manager signature verification failed.'
        }
        $managerVersion = & $installedManager --version 2>&1
        if ($LASTEXITCODE -ne 0 -or ($managerVersion -join "`n") -ne '7.2.20r175154') {
            throw 'Installed manager did not report the exact reviewed version.'
        }
        $attempt.installed_manager = $installedManager
        $attempt.installed_manager_sha256 = (Get-FileHash -LiteralPath $installedManager -Algorithm SHA256).Hash.ToLowerInvariant()
        $attempt.version_output = $managerVersion -join "`n"
        $attempt.installation_verified = $true
    } catch {
        $attempt.error_type = $_.Exception.GetType().Name
        if ($_.Exception.PSObject.Properties.Name -contains 'NativeErrorCode') {
            $attempt.native_error_code = $_.Exception.NativeErrorCode
        }
        throw
    } finally {
        $attempt.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
        $attempt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $attemptPath -Encoding utf8
        $attempt | ConvertTo-Json -Depth 4
    }
}
