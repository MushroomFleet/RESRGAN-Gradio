@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt
pip install basicsr
pip install facexlib
pip install gfpgan
pip install gradio==4.44.1

echo Setting up Real-ESRGAN...
python setup.py develop

echo Installation complete!
echo To activate the environment, run: venv\Scripts\activate.bat
