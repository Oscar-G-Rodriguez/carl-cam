; Directory-based, per-user development installer. No OBS/driver/Python install.
; Compile only after tools/verify_bundle.py confirms a camera-free self-test.
#ifndef BundleDir
  #define BundleDir "..\dist\Carl-Cam.dist"
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0-dev"
#endif
#ifndef CarlCamAppId
  #define CarlCamAppId "{{C10BE185-6F59-4D19-9B14-483B2391E3A4}"
#endif
#ifndef CarlCamInstallDir
  #define CarlCamInstallDir "{localappdata}\Programs\Carl-Cam"
#endif
#ifndef CarlCamGroup
  #define CarlCamGroup "Carl-Cam"
#endif

[Setup]
AppId={#CarlCamAppId}
AppName=Carl-Cam
AppVersion={#AppVersion}
AppPublisher=Carl-Cam contributors
DefaultDirName={#CarlCamInstallDir}
DefaultGroupName={#CarlCamGroup}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.22000
OutputDir=..\dist\installer
OutputBaseFilename=Carl-Cam-{#AppVersion}-Windows-x64-Setup
SetupIconFile=..\src\carl_cam\assets\graphics\carl-cam.ico
UninstallDisplayIcon={app}\Carl-Cam.exe
WizardStyle=modern
DisableProgramGroupPage=yes
DisableDirPage=no
Compression=lzma2
SolidCompression=yes
CloseApplications=no
RestartApplications=no
InfoBeforeFile=before-install.txt

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Carl-Cam"; Filename: "{app}\Carl-Cam.exe"; WorkingDir: "{app}"
Name: "{group}\Carl-Cam Quick Start"; Filename: "{app}\help\Quick Start.html"
Name: "{autodesktop}\Carl-Cam"; Filename: "{app}\Carl-Cam.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Carl-Cam.exe"; Description: "Open Carl-Cam"; Flags: nowait postinstall skipifsilent unchecked

; No [UninstallDelete] entry: users' numerical recordings, models and settings
; under LocalAppData\CarlCam are deliberately preserved by uninstall.
