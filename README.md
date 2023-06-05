# Flower Images Classification using CNN

## COSC2753 - Machine Learning
### Machine Learning Project: Assignment 2
### Group T3_G06

## Libraries Requirement
- pillow
- torch
- numpy
- pandas
- torchvision
- pickle
- tqdm
- matplotlib
- annoy

## Setup
How to run this project: 

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
