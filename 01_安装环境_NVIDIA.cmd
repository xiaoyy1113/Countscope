@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -Mode gpu
if errorlevel 1 (
 echo Operation failed. See the message above.
 pause
 exit /b 1
)
pause
