@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem
rem copy_repo.bat - sync this repo with your USB/flash drive (both directions),
rem skipping .venv and Python caches. The drive is found by its exact volume
rem label, or you can give a destination/source directory manually.
rem
rem Usage:
rem   copy_repo.bat                       PUSH  repo -> [flash]\projects\neural-speech
rem   copy_repo.bat -r                    PULL  [flash]\projects\neural-speech -> this PC (overwrites repo files, keeps .venv)
rem   copy_repo.bat -p myfolder           use folder [flash]\myfolder instead of "projects"
rem   copy_repo.bat -d "E:\backups" -r    manual base dir in either direction
rem   copy_repo.bat -d "E:\backups" -f    push to a manual dir, overwrite if exists
rem
rem Flags:
rem   -r              reverse (pull from flash to PC); default is push (PC to flash)
rem   -d BASE_DIR     use this directory as the base instead of the detected flash drive
rem   -p FOLDER       name of the folder on the base (default "projects")
rem   -f              overwrite the destination even if it already exists
rem
rem Defaults (edit these as you like, or set them as environment variables):
if not defined FLASH_LABEL set "FLASH_LABEL=KINGSTONE E"
if not defined DEST_PARENT_NAME set "DEST_PARENT_NAME=projects"
if not defined DEST_FOLDER_NAME set "DEST_FOLDER_NAME=neural-speech"

set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
set "REVERSE=0"
set "FORCE=0"
set "MANUAL="
set "PARENT=%DEST_PARENT_NAME%"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="-r" (set "REVERSE=1" & shift & goto parse)
if /i "%~1"=="-f" (set "FORCE=1" & shift & goto parse)
if /i "%~1"=="-d" (set "MANUAL=%~2" & shift & shift & goto parse)
if /i "%~1"=="-p" (set "PARENT=%~2" & shift & shift & goto parse)
goto usage
:parsed

if defined MANUAL (
  set "BASE=%MANUAL%"
) else (
  call :detect_flash
  if not defined DRIVE (
    echo ERROR: no removable drive labelled '%FLASH_LABEL%' was found. 1>&2
    echo Removable drives present: 1>&2
    powershell -NoProfile -Command "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=2' | Select-Object DeviceID,VolumeName,Size | Format-Table -AutoSize" 1>&2
    exit /b 1
  )
  set "BASE=!DRIVE!\!PARENT!"
)
rem strip trailing backslash from BASE
if "!BASE:~-1!"=="\" set "BASE=!BASE:~0,-1!"

set "REMOTE=!BASE!\%DEST_FOLDER_NAME%"

if "%REVERSE%"=="0" goto push
goto pull

:push
set "TARGET=!REMOTE!"
echo PUSH: repo -^> !REMOTE!
if exist "!TARGET!" if "%FORCE%"=="0" (
  echo ERROR: '!REMOTE!' already exists - re-run with -f to overwrite. 1>&2
  exit /b 1
)
if not exist "!BASE!" mkdir "!BASE!"
if exist "!TARGET!" rmdir /s /q "!TARGET!"
mkdir "!TARGET!"
call :copy_tree "%REPO%" "!TARGET!"
if errorlevel 8 (
  echo ERROR: copy failed. 1>&2
  exit /b 1
)
goto verify

:pull
set "TARGET=%REPO%"
echo PULL: !REMOTE! -^> repo ^(%REPO%^)
if not exist "!REMOTE!\" (
  echo ERROR: '!REMOTE!' does not exist on the flash drive. 1>&2
  exit /b 1
)
call :copy_tree "!REMOTE!" "!TARGET!"
if errorlevel 8 (
  echo ERROR: copy failed. 1>&2
  exit /b 1
)
goto verify

:verify
echo Verifying against !TARGET! ...
set "OK=1"
for %%P in (ns.py commands.txt checkpoints\mms\v026 checkpoints\mms\v030) do (
  if exist "!TARGET!\%%P" (
    echo   OK       %%P
  ) else (
    echo   MISSING  %%P 1>&2
    set "OK=0"
  )
)
if "%REVERSE%"=="0" (
  if exist "!TARGET!\.venv" (
    echo   FAIL  .venv present in copy 1>&2
    set "OK=0"
  ) else (
    echo   OK       .venv excluded
  )
) else (
  echo   OK       local .venv kept untouched
)
if "!OK!"=="0" exit /b 1

if "%REVERSE%"=="0" (
  powershell -NoProfile -Command "$s=(Get-ChildItem -LiteralPath '!TARGET!' -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum; Write-Host ('Done. Copied size: {0:N1} MB' -f ($s/1MB))"
  echo To pull back later: copy_repo.bat -r
) else (
  echo Done. Restored from flash ^(local .venv kept^).
)
exit /b 0

rem ---------------------------------------------------------------------------
:copy_tree
rem %~1 = source, %~2 = destination. robocopy exit codes below 8 mean success.
robocopy "%~1" "%~2" /E /NFL /NDL /NP /NJH ^
  /XD .venv .pytest_cache .mypy_cache .ruff_cache __pycache__ ^
  /XF *.pyc *.pyo *.pyd
exit /b %ERRORLEVEL%

:detect_flash
set "DRIVE="
for /f "delims=" %%D in ('powershell -NoProfile -Command "@(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=2').Where({$_.VolumeName -eq '%FLASH_LABEL%'}).DeviceID"') do (
  if not defined DRIVE set "DRIVE=%%D"
)
exit /b 0

:usage
for /f "usebackq skip=2 tokens=* delims=" %%L in ("%~f0") do (
  set "LINE=%%L"
  if /i "!LINE:~0,3!"=="rem" (echo(!LINE:~4!) else (exit /b 1)
)
exit /b 1
