import os
import json
import subprocess
import gradio as gr
import shutil
import time
import glob
import yaml

# Load configuration
def load_config():
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            return json.load(f)
    return {
        "dataset": {
            "input_path": "datasets/custom/input_images",
            "output_path": "datasets/custom/processed_images",
            "meta_info_path": "datasets/custom/meta_info/meta_info_custom.txt"
        },
        "training": {
            "model_name": "custom_RealESRGAN",
            "scale": 4,
            "gt_size": 256,
            "batch_size": 12,
            "pretrain_model_g": "experiments/pretrained_models/RealESRGAN_x4plus.pth",
            "pretrain_model_d": "experiments/pretrained_models/RealESRGAN_x4plus_netD.pth",
            "total_iter": 100000,
            "save_checkpoint_freq": 5000
        },
        "paths": {
            "output_dir": "experiments/custom_training"
        }
    }

# Save configuration
def save_config(config):
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

# Installation function
def install_environment():
    output = ""

    # Check if venv already exists
    if os.path.exists("venv"):
        output += "Virtual environment already exists. Skipping creation.\n"
    else:
        output += "Creating virtual environment...\n"
        subprocess.run(["python", "-m", "venv", "venv"], capture_output=True, text=True)

    # Activate venv and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "call venv\\Scripts\\activate.bat && "
    else:  # Linux/Mac
        activate_cmd = "source venv/bin/activate && "

    output += "Installing dependencies...\n"
    cmds = [
        f"{activate_cmd} pip install -r requirements.txt",
        f"{activate_cmd} pip install basicsr",
        f"{activate_cmd} pip install facexlib",
        f"{activate_cmd} pip install gfpgan",
        f"{activate_cmd} pip install gradio==4.0.0",
        f"{activate_cmd} python setup.py develop"
    ]

    for cmd in cmds:
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output += f"Command: {cmd}\n"
        output += process.stdout + "\n"
        if process.returncode != 0:
            output += f"Error: {process.stderr}\n"
            return output + "Installation failed!"

    return output + "Installation completed successfully!"

# Prepare data function
def prepare_data(input_path, output_path):
    config = load_config()
    config["dataset"]["input_path"] = input_path
    config["dataset"]["output_path"] = output_path
    save_config(config)

    output = ""

    # Create directories
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.dirname(config["dataset"]["meta_info_path"]), exist_ok=True)

    # Check if input directory has images
    image_files = glob.glob(os.path.join(input_path, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        return "Error: No image files found in the input directory. Please add images first."

    output += f"Found {len(image_files)} images in the input directory.\n"

    # Run multi-scale image generation
    output += "Generating multi-scale images...\n"
    process = subprocess.run(
        ["python", "scripts/generate_multiscale_DF2K.py", "--input", input_path, "--output", output_path],
        capture_output=True, text=True
    )
    output += process.stdout + "\n"

    if process.returncode != 0:
        output += f"Error during multi-scale generation: {process.stderr}\n"
        return output + "Data preparation failed!"

    # Run meta info generation
    output += "Generating meta information...\n"
    process = subprocess.run(
        [
            "python", "scripts/generate_meta_info.py",
            "--input", input_path, output_path,
            "--root", os.path.dirname(input_path), os.path.dirname(output_path),
            "--meta_info", config["dataset"]["meta_info_path"]
        ],
        capture_output=True, text=True
    )
    output += process.stdout + "\n"

    if process.returncode != 0:
        output += f"Error during meta info generation: {process.stderr}\n"
        return output + "Data preparation failed!"

    return output + "Data preparation completed successfully!"

# Train function
def train_model(model_name, total_iter, save_freq):
    config = load_config()
    config["training"]["model_name"] = model_name
    config["training"]["total_iter"] = int(total_iter)
    config["training"]["save_checkpoint_freq"] = int(save_freq)
    save_config(config)

    output = ""

    # Check if meta info exists
    if not os.path.exists(config["dataset"]["meta_info_path"]):
        return "Error: Meta information file does not exist. Please run data preparation first."

    # Create custom YAML configuration
    output += "Creating custom training configuration...\n"
    try:
        with open('options/finetune_realesrgan_x4plus.yml', 'r') as f:
            base_config = yaml.safe_load(f)

        base_config['name'] = config['training']['model_name']
        base_config['datasets']['train']['dataroot_gt'] = os.path.dirname(config['dataset']['input_path'])
        base_config['datasets']['train']['meta_info'] = config['dataset']['meta_info_path']
        base_config['path']['pretrain_network_g'] = config['training']['pretrain_model_g']
        base_config['path']['pretrain_network_d'] = config['training']['pretrain_model_d']
        base_config['train']['total_iter'] = config['training']['total_iter']
        base_config['logger']['save_checkpoint_freq'] = config['training']['save_checkpoint_freq']

        os.makedirs(os.path.dirname('options/custom_train.yml'), exist_ok=True)
        with open('options/custom_train.yml', 'w') as f:
            yaml.safe_dump(base_config, f)

        output += "Custom configuration created successfully.\n"

    except Exception as e:
        output += f"Error creating custom configuration: {str(e)}\n"
        return output + "Training failed!"

    # Run training
    output += "Starting training process. This may take a long time...\n"
    output += "Check the console for real-time training progress.\n"

    # Start training in a separate process
    process = subprocess.Popen(
        ["python", "realesrgan/train.py", "-opt", "options/custom_train.yml", "--auto_resume"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Display initial output
    start_time = time.time()
    line_count = 0
    initial_output = ""

    while time.time() - start_time < 5 and line_count < 20:  # Show initial 5 seconds or 20 lines
        line = process.stdout.readline()
        if not line:
            break
        initial_output += line
        line_count += 1

    return output + f"Training process started with PID {process.pid}.\n\nInitial output:\n{initial_output}\n\nTraining will continue in the background."

# Inference function
def inference(input_image, model_path, scale):
    if input_image is None:
        return "Please upload an image for inference.", None

    # Save input image to a temporary file
    temp_input = "temp_input.png"
    input_image.save(temp_input)

    # Run inference
    temp_output = "temp_output.png"

    if model_path and os.path.exists(model_path):
        cmd = ["python", "inference_realesrgan.py", "-n", "RealESRGAN_x4plus", "-i", temp_input, "-o", temp_output, "--outscale", str(scale), "--model_path", model_path]
    else:
        cmd = ["python", "inference_realesrgan.py", "-n", "RealESRGAN_x4plus", "-i", temp_input, "-o", temp_output, "--outscale", str(scale)]

    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.returncode != 0:
        return f"Error during inference: {process.stderr}", None

    # Return the output image
    if os.path.exists(temp_output):
        return "Inference completed successfully!", temp_output
    else:
        return "Error: Output image not found.", None

# Check and list available trained models
def list_trained_models():
    models = []

    # Check experiments directory
    experiment_dirs = glob.glob("experiments/**/models", recursive=True)
    for exp_dir in experiment_dirs:
        model_files = glob.glob(os.path.join(exp_dir, "net_g_*.pth"))
        for model_file in model_files:
            models.append(model_file)

    # Also check pre-trained models
    pretrained_dir = "experiments/pretrained_models"
    if os.path.exists(pretrained_dir):
        pretrained_models = glob.glob(os.path.join(pretrained_dir, "*.pth"))
        for model in pretrained_models:
            models.append(model)

    return models

# UI Components
with gr.Blocks(title="Real-ESRGAN Trainer") as app:
    gr.Markdown("# Real-ESRGAN Training Interface")

    with gr.Tabs():
        with gr.TabItem("Installation"):
            install_button = gr.Button("Install Environment")
            install_output = gr.Textbox(label="Installation Output", lines=10)
            install_button.click(fn=install_environment, outputs=install_output)

        with gr.TabItem("Data Preparation"):
            input_path = gr.Textbox(label="Input Images Path", value="datasets/custom/input_images")
            output_path = gr.Textbox(label="Processed Images Path", value="datasets/custom/processed_images")
            prepare_button = gr.Button("Prepare Dataset")
            prepare_output = gr.Textbox(label="Preparation Output", lines=10)

            prepare_button.click(fn=prepare_data, inputs=[input_path, output_path], outputs=prepare_output)

        with gr.TabItem("Training"):
            model_name = gr.Textbox(label="Model Name", value="custom_RealESRGAN")
            total_iter = gr.Number(label="Total Iterations", value=100000)
            save_freq = gr.Number(label="Save Checkpoint Frequency", value=5000)

            train_button = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Output", lines=10)

            train_button.click(fn=train_model, inputs=[model_name, total_iter, save_freq], outputs=train_output)

        with gr.TabItem("Inference"):
            with gr.Row():
                input_img = gr.Image(label="Input Image")
                output_img = gr.Image(label="Output Image")

            models = list_trained_models()
            model_dropdown = gr.Dropdown(label="Model", choices=models, value=models[0] if models else None)
            scale_slider = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Scale Factor")

            refresh_button = gr.Button("Refresh Model List")
            infer_button = gr.Button("Run Inference")
            infer_output = gr.Textbox(label="Inference Output")

            refresh_button.click(fn=list_trained_models, outputs=model_dropdown)
            infer_button.click(fn=inference, inputs=[input_img, model_dropdown, scale_slider], outputs=[infer_output, output_img])

if __name__ == "__main__":
    app.launch()
