# Flower Images Classification using CNN

## COSC2753 - Machine Learning
### Machine Learning Project: Assignment 2
### Group T3_G06

Instruction how to set up the environment to run our file

**To run the code successfully, users are required to install several libraries and use some modules: pillow, torch, torchvision, pickle, numpy, annoy, tqdm, matplotlib, pandas. Here are the steps to install them:
1. Open the terminal or command prompt
2. use pip statement and run the command correctly as we provided:
- pillow: pip install pillow
- torch: pip install torch
- numpy: pip install numpy
- pandas: pip install pandas
- torchvision: pip install torchvision
- pickle: you can directly use pickle by finding the code in the Notebook that contains "import pickle" and reun the code
- tqdm: pip install tqdm
- matplotlib: pip install matplotlib
- annoy: pip install annoy (if faces error, please see part 4)

3. Install 'annoy':
- First, install Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Then, run the VS Build Tool and install: Microsoft Visual C++ 14.0 or greater 
- Restart the PC
- Go to command prompt and install 'annoy': pip instll annoy

**How to run models**
1. Download the zip file and unzip the file
2. To run the model for task 1 and task 2, open and run from Command Prompt

* For task 1 *
- Open the Task 1 folder to get the location of Task_1.py file inside Task 1 folder
- Use "cd path/to/directory" command in your Command Prompt to navigate and switch your current directory to Task 1 folder
- After your directory in Command Prompt is pointed to Task 1 folder, type the below command and run:
	python "Task 1.py" "path/to/your_test_image_folder/your_chosen_test_image"
For example: python "Task 1.py" "C:\Users\ASUS STRIX\Documents\Machine Learning\assignment\Assignment2\Test\5. Pingpong\IMG_0627.PNG"
- Then wait and the result of classification will be displayed in a few second

* For task 2*
- Change directory to folder "Task 2". (cd path/to/directory)
- Use command: 
	python "Task 2.py" "path/to/your_test_image_folder/your_chosen_test_image"
For example: python "Task 2.py" "C:\Users\ASUS STRIX\Documents\Machine Learning\assignment\Assignment2\Test\5. Pingpong\IMG_0627.PNG"
- After finish running, go to folder "Task 2" to find file "reverse_search_output.png" to see the result.

**For notebooks**
- To run the notebook, please check if CUDA is available by this command:
	print(torch.cuda.is_available())
- If CUDA is not available, please install:
	pip install torch==<version>+cu<cuda_version> torchvision torchaudio -f https://download.pytorch.org/whl/cu<cuda_version>/torch_stable.html
- Or you can install Nvidia CUDA via this link:
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=Server2022&target_type=exe_local
