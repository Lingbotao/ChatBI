@echo off
chcp 437 >nul
echo Installing dependencies for Stock Analysis Assistant...
echo.

REM Check if pip is installed
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip not found, please ensure Python is installed and added to PATH environment variable.
    pause
    exit /b 1
)

REM Upgrade pip and install dependencies using domestic mirror for better speed
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Warning: Failed to upgrade pip, will continue installing dependencies
)

REM Install all dependencies from requirements.txt
echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
if errorlevel 1 (
    echo Trying to install with default source...
    python -m pip install -r requirements.txt
)

if errorlevel 1 (
    echo Installation failed, please check requirements.txt file and network connection.
    pause
    exit /b 1
)

echo.
echo Dependencies installation completed!
echo.

REM Check if key libraries are installed correctly
python -c "import dashscope; print('dashscope version:', dashscope.__version__)" >nul 2>&1
if errorlevel 1 (
    echo Warning: dashscope not installed correctly
) else (
    echo dashscope installed correctly
)

python -c "import pandas; print('pandas version:', pandas.__version__)" >nul 2>&1
if errorlevel 1 (
    echo Warning: pandas not installed correctly
) else (
    echo pandas installed correctly
)

python -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Warning: matplotlib not installed correctly
) else (
    echo matplotlib installed correctly
)

echo.
echo All dependencies have been installed successfully!
pause