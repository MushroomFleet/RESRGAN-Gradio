@echo off
call venv\Scripts\activate.bat

echo Reading training configuration...
python -c "import json; config = json.load(open('config.json')); print('Model name:', config['training']['model_name']); print('Scale:', config['training']['scale']); print('Total iterations:', config['training']['total_iter'])"

echo Creating custom YAML configuration...
python -c "import os, yaml, json; config = json.load(open('config.json')); os.makedirs(config['paths']['output_dir'], exist_ok=True); base_config = yaml.safe_load(open('options/finetune_realesrgan_x4plus.yml', 'r')); base_config['name'] = config['training']['model_name']; base_config['datasets']['train']['dataroot_gt'] = os.path.dirname(config['dataset']['input_path']); base_config['datasets']['train']['meta_info'] = config['dataset']['meta_info_path']; base_config['path']['pretrain_network_g'] = config['training']['pretrain_model_g']; base_config['path']['pretrain_network_d'] = config['training']['pretrain_model_d']; base_config['train']['total_iter'] = config['training']['total_iter']; base_config['logger']['save_checkpoint_freq'] = config['training']['save_checkpoint_freq']; os.makedirs(os.path.dirname('options/custom_train.yml'), exist_ok=True); yaml.safe_dump(base_config, open('options/custom_train.yml', 'w'))"

echo Starting training...
python realesrgan/train.py -opt options/custom_train.yml --auto_resume

echo Training complete!
