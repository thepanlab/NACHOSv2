# Command Line Basics

This guide provides an introduction to using the command line interface (CLI) for our image classification pipeline. If you're new to using command line tools, especially in a supercomputer environment, this section will help you get started.

## Accessing the Command Line

1. On the supercomputer, you typically access the command line through a terminal or SSH session.
2. Once connected, you'll see a prompt, usually ending with a `$` or `>` symbol.

## Essential Commands

Here are some basic commands you'll need to navigate and use our pipeline:

1. **Change Directory**:
   ```
   cd /path/to/pipeline
   ```
   Use this to navigate to the pipeline's directory.

2. **List Directory Contents**:
   ```
   ls
   ```
   This shows the files and folders in the current directory.

3. **Print Working Directory**:
   ```
   pwd
   ```
   This shows your current location in the file system.

4. **Activate Conda Environment**:
   ```
   conda activate pytorch_gpu_env
   ```
   Always activate the environment before running pipeline commands.

## Running the Pipeline

To run the pipeline, you'll typically use a command in this format:

```
python3 -m scrpits.training.training_sequential.training_inner_loop.py [options]
```

Replace `[options]` with specific arguments for your run.

## Common Pipeline Commands

Here are some example commands you might use with our pipeline:

1. **Start Training**:
   ```
   python3 -m scripts.training.training_sequential.training_inner_loop --file scripts/config_files/3D_config_inner_conv3D_trial_parallel.json
   ```

2. **Process Results**:
   ```
   python3 src/results_processing/learning_curve/learning_curve_many.py
   ```

Note: These are example commands. Refer to the [Using the Pipeline](using_the_pipeline.md) section for specific commands and options.

## Tips for Command Line Use

1. Use the up and down arrow keys to navigate through your command history.
2. Use Tab for auto-completion of file and directory names.
3. If you need to stop a running process, use Ctrl+C.
4. To clear the screen, use the `clear` command.

## Next Steps

Once you're comfortable with these basics, proceed to the [Using the Pipeline](using_the_pipeline.md) section for detailed instructions on running specific pipeline tasks.

Remember, practice makes perfect. Don't hesitate to experiment with these commands in a safe directory to get more comfortable with the command line interface.