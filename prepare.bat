@echo off
call venv\Scripts\activate.bat

echo Reading configuration...
python -c "import json; config = json.load(open('config.json')); print('Using input path:', config['dataset']['input_path']); print('Output path:', config['dataset']['output_path']); print('Meta info path:', config['dataset']['meta_info_path'])"

echo Creating necessary directories...
python -c "import os, json; config = json.load(open('config.json')); os.makedirs(config['dataset']['input_path'], exist_ok=True); os.makedirs(config['dataset']['output_path'], exist_ok=True); os.makedirs(os.path.dirname(config['dataset']['meta_info_path']), exist_ok=True)"

echo Generating multi-scale images...
python -c "import json, os, subprocess; config = json.load(open('config.json')); subprocess.run(['python', 'scripts/generate_multiscale_DF2K.py', '--input', config['dataset']['input_path'], '--output', config['dataset']['output_path']])"

echo Generating meta information...
python -c "import json, os, subprocess; config = json.load(open('config.json')); subprocess.run(['python', 'scripts/generate_meta_info.py', '--input', config['dataset']['input_path'], config['dataset']['output_path'], '--root', os.path.dirname(config['dataset']['input_path']), os.path.dirname(config['dataset']['output_path']), '--meta_info', config['dataset']['meta_info_path']])"

echo Data preparation complete!
