@echo off
REM Mix Studio Launcher for Windows
REM Sets PyTorch scratch disk to E: drive and launches the server

setlocal EnableDelayedExpansion

REM === Scratch Disk Configuration ===
REM Redirect PyTorch temporary files to E: drive instead of default D:
if not exist "E:\tmp\pytorch" mkdir "E:\tmp\pytorch" 2>nul

REM Set environment variables for this process and its children
set "TEMP=E:\tmp\pytorch"
set "TMP=E:\tmp\pytorch"

REM === Node.js Configuration ===
set "NODE_EXE="

REM Try standard PATH lookup first (works when node is in system PATH)
for %%N in (node) do set "NODE_EXE=%%~$PATH:N"

REM Fallback: check common non-standard install locations
if not defined NODE_EXE if exist "%LOCALAPPDATA%\Programs\Nodejs\node.exe" set "NODE_EXE=%LOCALAPPDATA%\Programs\Nodejs\node.exe"
if not defined NODE_EXE if exist "%ProgramFiles%\nodejs\node.exe" set "NODE_EXE=%ProgramFiles%\nodejs\node.exe"
if not defined NODE_EXE if exist "%USERPROFILE%\.hermes\node\node.exe" set "NODE_EXE=%USERPROFILE%\.hermes\node\node.exe"

REM Last resort: use 'where' command to search PATH more thoroughly
if not defined NODE_EXE for /f "delims=" %%P in ('where node 2^>nul') do (set "NODE_EXE=%%P" & goto :nodeFound)
:nodeFound

if not defined NODE_EXE (
    echo [Mix Studio] Error: Node.js was not found. Install from https://nodejs.org/ and run this file again.
    pause
    exit /b 1
)

REM === Version Check ===
for /f "tokens=1 delims=. " %%V in ('%NODE_EXE% -p "process.version"') do set "NODE_MAJOR=%%V"

if not defined NODE_MAJOR (
    echo [Mix Studio] Error: Could not determine Node.js version.
    pause
    exit /b 1
)

REM Strip the leading 'v' from version string (e.g., v22 -> 22)
set "NODE_MAJOR=%NODE_MAJOR:v=%"

if "%NODE_MAJOR%" LSS "22" (
    echo [Mix Studio] Error: Mix Studio requires Node.js 22 or newer. Install the current LTS from https://nodejs.org/
    pause
    exit /b 1
)

REM === Check Installation Status ===
if not exist "install.json" (
    echo [Mix Studio] Running initial setup...
    "%NODE_EXE%" installer/bootstrap.js || exit /b %ERRORLEVEL%
)

REM === Launch Server ===
echo [Mix Studio] Starting server on port !PORT!...
set "MIXBOX_RESTART_MODE=launcher"
"%NODE_EXE%" server.js

endlocal