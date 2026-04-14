@echo off
:: Set console to UTF-8 to handle the arrows and symbols
chcp 65001 > nul

set LOGFILE=pipeline_log.txt
set PY_EXE="C:\Users\koden\AppData\Local\Microsoft\WindowsApps\python.exe"

echo ====================================================== >> %LOGFILE%
echo Pipeline Started: %date% %time% >> %LOGFILE%
echo ====================================================== >> %LOGFILE%

echo [1/8] Running Step 09: Targeted Panel...
%PY_EXE% step09_targeted_pannel.py
if %ERRORLEVEL% neq 0 goto :misfire

echo [2/8] Running Step 09a: ICP Discovery...
%PY_EXE% step09a_icp_discovery.py
if %ERRORLEVEL% neq 0 goto :misfire

echo [3/8] Running Step 09b: DML Effects...
%PY_EXE% step09b_dml_effects.py
if %ERRORLEVEL% neq 0 goto :misfire

echo [4/8] Running Step 10: Rebuilding Temporal KG...
%PY_EXE% step10_build_temporal_kg.py
if %ERRORLEVEL% neq 0 goto :misfire

echo [5/8] Running Step 11: Building Causal KG...
%PY_EXE% step11_build_causal_kg.py
if %ERRORLEVEL% neq 0 goto :misfire

echo [6/8] Running Step 11b: Adding ICP/DML Edges...
%PY_EXE% step11b_add_icp_dml_to_kg.py
if %ERRORLEVEL% neq 0 goto :misfire

echo [7/8] Running Step 12: Intrinsic Evaluation...
%PY_EXE% step12_intrinsic_evaluation.py
if %ERRORLEVEL% neq 0 goto :misfire

echo [8/8] Running Step 12b: Novel Metrics...
%PY_EXE% step12b_novel_metrics.py
if %ERRORLEVEL% neq 0 goto :misfire

echo.
echo ======================================================
echo ALL STEPS COMPLETED SUCCESSFULLY
echo ======================================================
echo Finished at: %date% %time% >> %LOGFILE%
pause
exit /b 0

:misfire
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo FATAL ERROR: The last script crashed. Pipeline stopped.
echo Check the console output above for details.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo FAILED at: %date% %time% >> %LOGFILE%
pause
exit /b 1