# Classifying MNIST dataset using CNTK

This sample uses CNTK to create a multi-layer neural network to classify MNIST dataset.

The code in this sample is adapted from the following CNTK tutorials:
1. https://github.com/Microsoft/CNTK/blob/v2.0/Tutorials/CNTK_103A_MNIST_DataLoader.ipynb
2. https://github.com/Microsoft/CNTK/blob/v2.0/Tutorials/CNTK_103C_MNIST_MultiLayerPerceptron.ipynb


## Instructions for running the script from CLI window
You can run your scripts from the Workbench app. However, we use the command-line window to watch the feedback in real time.

### Running your CNTK-based script locally
Open the command-line window by clicking on File --> Open Command Prompt and install the right CNTK version for your platform. You can refer to this article for setting up CNTK: [Setup CNTK on your machine](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine)

Once you select the right wheel file for your platform, you can install it locally by running the following command in your CLI command prompt.
```
# You only need to do this once. This whl file is for Windows operating system.
$ pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.1-cp35-cp35m-win_amd64.whl
```

Once you install CNTK, you can now run your script using the following command: 
```
# submit the experiment to local execution environment
$ az ml experiment submit -c local cntk_mnist.py
```

### Running your CNTK script on local or remote Docker
If you have a Docker engine running locally, you can run `cntk_mnist.py` in a local docker container. Since Docker-based runs are managed by **conda_dependencies.yml** file, it needs to have a reference to the right whl file. **conda_dependencies.yml** for this sample already has that reference.
```
dependencies:
    - pip:
      - https://cntk.ai/PythonWheel/CPU-Only/cntk-2.1-cp35-cp35m-linux_x86_64.whl
```

Run the following command for executing your script on local Docker:
```
# submit the experiment to local Docker container for execution
$ az ml experiment submit -c docker cntk_mnist.py
```

You can also execute your script on Docker on a remote machine. Similar to local Docker execution, **conda_dependencies.yml** needs to have the following reference:
```
dependencies:
    - pip:
      - https://cntk.ai/PythonWheel/CPU-Only/cntk-2.1-cp35-cp35m-linux_x86_64.whl
```
If you have a compute target named _myvm_ for a remote VM, you can run the following command to execute your script:

```
$ az ml experiment submit -c myvm cntk_mnist.py
```


>[!NOTE] Your first execution on docker-based compute target automatically downloads a base Docker image. For that reason, it takes a few minutes before your job starts to run. Your environment is then cached to make subsequent runs faster. 

## Running your CNTK script on a VM with GPU
You can get a huge performance boost by running computationally intensive tasks such as neural-network training a machine with GPUs.

>[!NOTE] If your local machine already has NVidia GPU chips and you have installed the CUDA libraries and toolkits, you can directly run the script using local compute target. The following instructions are specifically for running scripts in a remote VM equipped with GPU.

### Step 1. Provision Linux VM with GPU
Create an Ubuntu-based Data Science Virtual Machine(DSVM) in Azure portal using one of the NC-series VM templates. NC-series VMs are the VMs equipped with GPUs for computation.

### Step 2. Attach the compute target
Run following command to add the GPU VM as a compute target in your current project:
```
$ az ml computetarget attach --name myvm --address <ip address or FQDN> --username <username> --password <pwd> --type remotedocker
```
This command creates a `myvm.compute` and `myvm.runconfig` files under the `aml_config` folder.

### Step 3. Modify the configuration files under _aml_config_ folder
- Install GPU-based version of CNTK by adding the following dependency in your **conda_dependencies.yml** file.
```
dependencies:

  - pip:
    - https://cntk.ai/PythonWheel/GPU/cntk-2.1-cp35-cp35m-linux_x86_64.whl
```
[!IMPORTANT] Replace the existing reference in your conda_dependecies.yml file that refers to CNTK library for CPUs.

- Replace the value of `baseImage` from `microsoft/mmlspark:plus-0.7.91` to  `microsoft/mmlspark:plus-gpu-0.7.91` in your `myvm.compute` file. 

- Add a line `nvidiaDocker: true` in your `myvm.compute`.

- Change the value of `Framework` from `PySpark` to `Python` in your `myvm.runconfig` file


### Step 4. Run the script.
Now you are ready to run the script.
```
$ az ml experiment submit -c myvm cntk_mnist.py
```
You notice that the script finishes faster than using CPU. The command-line outputs should indicate that GPU is used for executing this script.
