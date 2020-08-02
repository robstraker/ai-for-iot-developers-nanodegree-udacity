# Smart Queue Monitoring System
Build custom queuing systems for the retail, manufacturing and transportation sectors and use the Intel® DevCloud for the Edge to test your solutions performance.

## Proposal Submission
For each of the industry sector scenarios, this proposal lists the hardware selection, client requirements, and rationale for selection:

### Manufacturing: 
- **Hardware Selection:** 
  - CPU + FPGA
- **Client Requirements:**
  - Power: The client is moving towards an energy efficient workplace.
  - Space: There are no stated space constraints, but it is likely that it is preferable that devices be added to existing computers.
  - Economic: There are no stated constraints related to costs, so cost does not appear to be much of a factor.
  - Flexibility: The client wants a device that has future flexibility so they can reprogram and optimize the system for different use-cases.
  - Environment: The client wants a system that can operate in a manufacturing environment.
  - Lifespan: Ideally, the client would like the system to last at least 5-10 years.
  - Performance: The client would like the system to run inference on the video stream very quickly, so it can detect chip flaws without slowing packaging.
- **Rationale for Selection:** 
  - An FPGA device would make the most sense for this client. 
  - A removable NCS2 device would not be the best option, as it is to be installed on the factory floor, which is an open environment. 
  - A CPU or GPU would not provide the flexibility, robustness, lifespan, or performance desired. 
  - This leaves an FPGA card which would fit in a PCIe slot within the chassis of existing computers.

### Retail:
- **Hardware Selection:** 
  - CPU + Integrated GPU
- **Client Requirements:**
  - Power: The client wants to save as much as possible on his electric bill.
  - Space: The client does not have any store floor space available, and currently already has computers at each checkout counter.
  - Economic: The client does not have much money to invest in additional hardware.
  - Environment: The client wants a system that can operate in a retail environment.
  - Lifespan: The client has not expressed any requirements for lifespan, but presumably they would like a system that will not need to be replaced or upgraded for several years at a minimum.
  - Performance: The client has not specified particular requirements for performance, but the system will need to be able to monitor queues that form during waits at counters during the busiest times during the day.
- **Rationale for Selection:** 
  - A CPU + Integrated GPU device would make the most sense for the client.
  - The client wants to save on electricity costs, does not have any store floor space available, and does not have much money to invest.
  - However, he has personal computers installed with Intel i7 core processors for most checkout counters, so CPU + Integrated GPUs are already available.

### Transportation: 
- **Hardware Selection:** 
  - CPU + VPU
- **Client Requirements:**
  - Power: The client wants to save as much as possible on future power requirements.
  - Space: The client already has 7 CCTV cameras on the platform connected to PCs located in a nearby security booth, so there is not a lot of available space for new devices. 
  - Economic: The client has a maximum budget of $300 per machine, and would like to save as much as possible on hardware requirements.
  - Environment: The client wants a system that can operate in a busy urban passenger transportation environment.
  - Lifespan: The client has not expressed any requirements for lifespan, but presumably they would like a system that will not need to be replaced or upgraded for several years at a minimum.
  - Performance: The client has not specified particular requirements for performance, but the system will need to be able to monitor queues that form at train doors during the busiest times during the day.
- **Rationale for Selection:** 
  - A CPU + GPU option is not suitable, since Ms. Leah’s current PC system is used for CCTV recording, so there may not be enough capacity in this machine to perform inference.
  - Therefore, some kind of add-in accelerator card is recommended.
  - An FPGA card would cost over $1,000, which exceeds Ms. Leah’s budget of $100 to $150.
  - Therefore, one or two NCS2 sticks would be the best option.

## Project Set Up and Installation
This project was run on Udacity's workspace, so no additional setup was required.

Files were set up according to the following project directory structure: 
- **bin/queue_param** - Folder with queue paramaters.
  - manufacturing.npy
  - retail.npy
  - transportation.npy
- **demoTools** - Folder with various utilities.
  - catalog.css
  - catalog.py
  - checkStatus.sh
  - demoutils.py
  - OpenVINO_IoT_Examples_Cpp.json
  - OpenVINO_IoT_Examples_Python.json
  - progressUpdate.cpp
  - refreshRepository.sh
- **proposals** - Folder with proposals for each scenario.
  - proposal_outline.pdf
- **resources** - Folder with original videos for each scenario.
  - manufacturing.mp4
  - retail.mp4
  - transportation.mp4
- **results** - Folder with output logs, videos, and statistics.
  - Manufacturing:
      - cpu:
        - output_video.mp4
        - stats.txt
      - fpga:
        - output_video.mp4
        - stats.txt
      - gpu:
        - output_video.mp4
        - stats.txt
      - vpu:
        - output_video.mp4
        - stats.txt
  - Retail: organized as for Manufacturing
  - Transportation: organized as for Manufacturing  
  - output.tgz
  - stderr.log
  - stdout.log
- **src** - Folder with notebooks, scripts, and source code. 
  - Create_Job_Submission_Script.ipynb
  - Create_Python_Script.ipynb
  - people_deployment-manufacturing.ipynb
  - people_deployment-retail.ipynb
  - people_deployment-transportation.ipynb
  - person_detect.py
  - queue_job.sh
- instructions.md
- README.md

No models needed to be downloaded, as the `person-detection-retail-0013` model was available in the workspace. 

There were no dependencies required by this project.

## Documentation
There are five steps required to complete this project:

### Step 1: Create the Python Script

In the cells in the Create_Python_Script.ipynb notebook, you will need to complete the Python script and run the cell to generate the file using the magic `%%writefile` command. Your main task is to complete the following methods for the `PersonDetect` class:
- `load_model`
- `predict`
- `draw_outputs`
- `preprocess_outputs`
- `preprocess_inputs`

For your reference, here are all the arguments used for the argument parser in the command line:
- `--model`:  The file path of the pre-trained IR model, which has been pre-processed using the model optimizer. There is automated support built in this argument to support both FP32 and FP16 models targeting different hardware.
- `--device`: The type of hardware you want to load the model on (CPU, GPU, MYRIAD, HETERO:FPGA,CPU)
- `--video`: The file path of the input video.
- `--output_path`: The location where the output stats and video file with inference needs to be stored (results/[device]).
- `--max_people`: The max number of people in queue before directing a person to another queue.
- `--threshold`: The probability threshold value for the person detection. Optional arg; default value is 0.60.

### Step 2: Create Job Submission Script

The next step is to create our job submission script. In the cells in the Create_Job_Submission_Script.ipynb notebook, you will need to complete the job submission script and run the cell to generate the file using the magic `%%writefile` command. Your main task is to complete the following items of the script:

- Create a variable `MODEL` and assign it the value of the first argument passed to the job submission script.
- Create a variable `DEVICE` and assign it the value of the second argument passed to the job submission script.
- Create a variable `VIDEO` and assign it the value of the third argument passed to the job submission script.
- Create a variable `PEOPLE` and assign it the value of the sixth argument passed to the job submission script.

### Step 3: Smart Queue Monitoring System - Manufacturing Scenario

**Overview:**
Now that you have your Python script and job submission script, you're ready to request an **IEI Tank-870** edge node and run inference on the different hardware types (CPU, GPU, VPU, FPGA). After the inference is completed, the output video and stats files need to be retrieved and stored in the workspace, which can then be viewed within the Jupyter Notebook.

**Objectives:**
- Submit inference jobs to Intel's DevCloud using the `qsub` command.
- Retrieve and review the results.
- After testing, go back to the proposal doc and update your original proposed hardware device.

### Step 4: Smart Queue Monitoring System - Retail Scenario

**Overview:**
Now that you have your Python script and job submission script, you're ready to request an **IEI Tank-870** edge node and run inference on the different hardware types (CPU, GPU, VPU, FPGA). After the inference is completed, the output video and stats files need to be retrieved and stored in the workspace, which can then be viewed within the Jupyter Notebook.

**Objectives:**
- Submit inference jobs to Intel's DevCloud using the `qsub` command.
- Retrieve and review the results.
- After testing, go back to the proposal doc and update your original proposed hardware device.

### Step 5: Smart Queue Monitoring System - Transportation Scenario

**Overview:**
Now that you have your Python script and job submission script, you're ready to request an **IEI Tank-870** edge node and run inference on the different hardware types (CPU, GPU, VPU, FPGA). After the inference is completed, the output video and stats files need to be retrieved and stored in the workspace, which can then be viewed within the Jupyter Notebook.

**Objectives:**
* Submit inference jobs to Intel's DevCloud using the `qsub` command.
* Retrieve and review the results.
* After testing, go back to the proposal doc and update your original proposed hardware device.


## Results
Following are all the results generated by this project:

### Manufacturing: 
| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |  11.3                                        |  1.5                       |  FP16                   |
| GPU              |  11.4                                        |  60.5                      |  FP16                   |
| FPGA             |  9.0                                         |  29.1                      |  FP16                   |
| VPU              |  44.3                                        |  2.9                       |  FP16                   |

### Retail: 
| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |  4.5                                         |  1.7                       |  FP16                   |
| GPU              |  5.1                                         |  55.4                      |  FP16                   |
| FPGA             |  3.8                                         |  29.1                      |  FP16                   |
| VPU              |  25.1                                        |  2.8                       |  FP16                   |

### Transportation: 
| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |  16.6                                        |  1.4                       |  FP16                   |
| GPU              |  16.2                                        |  55.1                      |  FP16                   |
| FPGA             |  13.5                                        |  29.0                      |  FP16                   |
| VPU              |  50.3                                        |  2.8                       |  FP16                   |


## Conclusions
For all three scenarios:
- Average inference times were comparable for the CPU, FPGA, and GPU. However they were 5x as long for the VPU hardware.
- Model loading times were lowest for the CPU and VPU, approximately 15x longer for the FPGA, and approximately 30x longer for the GPU.

Alignment with hardware recommendations:
- Manufacturing:
  - Recommendation: CPU + FPGA
  - Performance requirement: To run inference on the video stream very quickly, so it can detect chip flaws without slowing packaging.
  - Results: FPGA achieves the lowest average time for inference.
  - Conclusion: Our original recommendation has been confirmed.
- Retail:
  - Recommendation: CPU + Integrated GPU
  - Performance requirement: System will need to be able to monitor queues that form during waits at counters during the busiest times during the day.
  - Results: GPU achieves one of the lowest average times for inference, although it takes the longest time to load.
  - Conclusion: Our original recommendation has been confirmed, as model needs to only be loaded once at the start of the day.
- Transportation:
  - Recommendation: CPU + VPU
  - Performance requirement: System will need to be able to monitor queues that form at train doors during the busiest times during the day.
  - Results: VPU achieves the highest average time for inference, but is one of the fastest to load the model. A 50 ms time for inference still allows for 20 measurements per second.
  - Conclusion: Our original recommendation has been confirmed.
