[app]

# title of your application
title = Carl-Cam

# project root directory. default = The parent directory of input_file
project_dir = .

# source file entry point path. default = main.py
input_file = Carl-Cam.py

# directory where the executable output is generated
exec_directory = dist

# path to the project file relative to project_dir
project_file = 

# application icon
icon = src/carl_cam/assets/graphics/carl-cam.ico

[python]

# python path
python_path = .venv/Scripts/python.exe

# python packages to install
packages = Nuitka==4.1.1

# buildozer = for deploying Android application
android_packages = buildozer==1.5.0,cython==0.29.33

[qt]

# paths to required qml files. comma separated
# normally all the qml files required by the project are added automatically
# design studio projects include the qml files using qt resources
qml_files = 

# excluded qml plugin binaries
excluded_qml_plugins = 

# qt modules used. comma separated
modules = Core,Gui,Svg,Widgets

# qt plugins used by the application. only relevant for desktop deployment
# for qt plugins used in android application see [android][plugins]
plugins = sensible

[android]

# path to pyside wheel
wheel_pyside = 

# path to shiboken wheel
wheel_shiboken = 

# plugins to be copied to libs folder of the packaged application. comma separated
plugins = 

[nuitka]

# usage description for permissions requested by the app as found in the info.plist file
# of the app bundle. comma separated
# eg = extra_args = --show-modules --follow-stdlib
macos.permissions = 

# mode of using nuitka. accepts standalone or onefile. default = onefile
mode = standalone

# specify any extra nuitka arguments
extra_args = --noinclude-qt-translations --windows-console-mode=disable --include-package=carl_cam --include-package=scipy._external.array_api_compat.numpy --include-data-dir=src/carl_cam/assets=carl_cam/assets --include-distribution-metadata=numpy --include-distribution-metadata=torch --include-distribution-metadata=scikit-learn --include-distribution-metadata=PySide6 --include-distribution-metadata=mediapipe --include-distribution-metadata=pyvirtualcam --include-distribution-metadata=cv2_enumerate_cameras --user-package-configuration-file=packaging/carl-cam.nuitka-package.config.yml --module-parameter=torch-disable-jit=yes --nofollow-import-to=torch.testing._internal.common_methods_invocations,torch.testing._internal.common_utils,torch.testing._internal.common_device_type,torch.testing._internal.common_modules,torch.testing._internal.common_nn,torch.testing._internal.opinfo*,torch._dynamo.test_case --assume-yes-for-downloads --mingw64 --jobs=2 --low-memory --report=build/nuitka-report.xml

[buildozer]

# build mode
# possible values = ["aarch64", "armv7a", "i686", "x86_64"]
# release creates a .aab, while debug creates a .apk
mode = debug

# path to pyside6 and shiboken6 recipe dir
recipe_dir = 

# path to extra qt android .jar files to be loaded by the application
jars_dir = 

# if empty, uses default ndk path downloaded by buildozer
ndk_path = 

# if empty, uses default sdk path downloaded by buildozer
sdk_path = 

# other libraries to be loaded at app startup. comma separated.
local_libs = 

# architecture of deployed platform
arch = 

