@echo off
setlocal
title He thong du doan ket qua hoc tap
cd /d "%~dp0"

echo ==================================================
echo   HE THONG DU DOAN KET QUA HOC TAP 
echo ==================================================
echo.

rem ---------- 1. Tim Python ----------
set "PY="
python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY (
    echo [LOI] Khong tim thay Python.
    echo       Hay cai Python 3.10 tro len tai https://www.python.org/downloads/
    echo       Nho tich vao o "Add python.exe to PATH" khi cai, roi chay lai file nay.
    echo.
    pause
    exit /b 1
)

rem ---------- 2. Tao moi truong ao (chi lan dau) ----------
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Tao moi truong ao .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [LOI] Khong tao duoc moi truong ao.
        pause
        exit /b 1
    )
)
call ".venv\Scripts\activate.bat"

rem ---------- 3. Cai thu vien (chi lan dau) ----------
if not exist ".venv\da_cai_thu_vien.flag" (
    echo [2/4] Cai thu vien, can Internet va mat vai phut ...
    python -m pip install --upgrade pip >nul 2>nul
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [LOI] Cai thu vien that bai. Kiem tra ket noi Internet roi chay lai.
        pause
        exit /b 1
    )
    echo da cai> ".venv\da_cai_thu_vien.flag"
)

rem ---------- 4. Chuan bi du lieu va mo hinh (chi khi chua co) ----------
echo [3/4] Kiem tra du lieu va mo hinh ...
if not exist "du_lieu\sinh_vien_v3.csv" python tao_du_lieu_v3.py
if not exist "mo_hinh_v3.pkl" python huan_luyen.py
if exist "du_lieu\student-mat.csv" if not exist "mo_hinh_du_lieu_that.pkl" python du_lieu_that.py

rem ---------- 5. Bo qua cau hoi email cua Streamlit lan dau ----------
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

rem ---------- 6. Khoi dong ung dung ----------
echo [4/4] Khoi dong ung dung. Trinh duyet se tu mo, neu khong hay vao http://localhost:8501
echo       De dung ung dung: nhan Ctrl+C hoac dong cua so nay.
echo.
python -m streamlit run app.py --browser.gatherUsageStats false
if errorlevel 1 (
    echo.
    echo [LOI] Ung dung dung bat thuong. Xem thong bao phia tren.
    pause
)
endlocal
