rmdir /s /q win_build
mkdir win_build
cd win_build

rem vcpkg remove catch2:x64-windows

vcpkg install catch2:x64-windows

set VCPKG_ROOT="C:\Users\ls04a\OneDrive\Documents\c++ tutorial\vcpkg\installed\x64-windows\share\catch2"
set MSBUILD="c:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
set UNIT_TESTS="C:\Users\ls04a\OneDrive\Documents\fast-ndarray-c++\win_build\Debug\fast_nd_array_in_cpp.exe"

cmake .. -G "Visual Studio 17 2022" ^
-A x64 -D Catch2_DIR=%VCPKG_ROOT%

%MSBUILD% ^
    fast_nd_array_in_cpp.sln /p:Configuration=Debug /p:Platform=x64

%UNIT_TESTS% --reporter tap
