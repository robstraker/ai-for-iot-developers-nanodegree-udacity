# Intel® Edge AI for IoT Developers Nanodegree
**In collaboration with Intel®**

Instructors: Stewart Christie, Micheal Virgo, Soham Chatterjee, Vaidheeswaran Archana. For biographical information, see [BIO.md][1]

Offered By: [udacity.com][2]

## Introduction
This repo contains projects and exercises for the four-part Intel® Edge AI for IoT Developers Nanodegree program offered through Udacity. Feel free to use the material for reference purposes, or if you get stuck. However, I would encourage you to try to complete all projects and exercises yourself, so that you can maximize your learning and enjoyment of the program.

## Udacity Description
#### About this Nanodegree
Lead the development of cutting-edge Edge AI applications for the future of the Internet of Things. 

Leverage the Intel® Distribution of OpenVINO™ Toolkit to fast-track development of high-performance computer vision & deep learning inference applications.

#### Overview
Edge AI applications are revolutionizing the IoT industry by bringing fast, intelligent behavior to the locations where it is needed. In this Nanodegree program, you will learn how to develop and optimize Edge AI systems, using the Intel® Distribution of OpenVINO™ Toolkit. A graduate of this program will be able to:
- Leverage the Intel® Distribution of OpenVINO™ Toolkit to fast-track development of high-performance computer vision and deep learning inference applications.
- Run pre-trained deep learning models for computer vision on-prem.
- Identify key hardware specifications of various hardware types (CPU, VPU, FPGA, and Integrated GPU). • Utilize Intel® DevCloud for the Edge to test model performance on various hardware types (CPU, VPU,
FPGA, and Integrated GPU).

This program consists of 3 courses and 3 projects. In each project you build will be an opportunity to demonstrate what you’ve learned in the course, and will demonstrate to potential employers that you have skills in these areas.

## Topics Covered
**Course 1: Edge AI Fundamentals with OpenVINO™:**  
Intel® Distribution of OpenVINOTM Toolkit, Pre-Trained Models, The Model Optimizer, The Inference Engine, MQTT

**Course 2: Hardware for Computer Vision Deep Learning Application Deployment:**  
CPUs, Integrated GPUs, Vision Processing Units (VPUs), Field Programmable Arrays (FPGAs), Intel® DevCloud for the Edge

**Course 3: Optimization Techniques and Tools for Computer Vision Deep Learning Applications:**  
Software Optimization, Intel® Deep Learning Workbench, VTune Amplifier

## Syllabus

### Course 1: Edge AI Fundamentals with OpenVINO™
Learn how to optimize your model and application code to reduce inference time when running your model at the edge. Use different software optimization techniques to improve the inference time of your model. Calculate how computationally expensive your model is. Use DL Workbench to optimize your model and benchmark the performance of your model. Use a VTune amplifier to find and fix hotspots in your application code. Finally, package your application code and data so that it can be easily deployed to multiple devices.

**Lesson 1: Leveraging Pre-Trained Models.** 
- Outcomes: 
  - Leverage a pre-trained model for computer vision inferencing. 
- Exercises:
  - [Loading Pre-Trained Models][3]
  - [Pre-processing Inputs][4]
  - [Deploy An App at the Edge][5]

**Lesson 2: The Model Optimizer.** 
- Outcomes: 
  - Convert pre-trained models into the framework agnostic intermediate representation with the Model Optimizer.
- Exercises:
  - [Convert a TF Model][6]
  - [Convert a Caffe Model][7]
  - [Convert an ONNX Model][8]
  - [Custom Layers][9]

**Lesson 3: The Inference Engine.** 
- Outcomes: 
  - Perform efficient inference on deep learning models through the hardware-agnostic Inference Engine.
- Exercises:
  - [Feed an IR to the Inference Engine][10]
  - [Inference Requests][11]
  - [Integrate into an App][12]

**Lesson 4: Deploying An Edge App.** 
- Outcomes: 
  - Deploy an app on the edge, including sending information through MQTT, and analyze model performance and use cases.
- Exercises:
  - [Handling Input Streams][13]
  - [Process Model Outputs][14]
  - [Server Communications][15]

**Project: [Deploy a People Counter App at the Edge][16].**
- In this project, you will utilize the Intel® Distribution of the OpenVINOTM Toolkit to build a People Counter app for inference at the edge. You will investigate different pre-trained models for person detection, and then convert the best model for optimized inference. The model will be deployed on the edge, such that only data on 1) the number of people in the frame, 2) time those people spent in frame, and 3) the total number of people counted are sent to a web server; inference will be done on the local machine.
- You will need to develop a method to compare the performance of your models before and after use of the Intel® Distribution of OpenVINOTM Toolkit for optimization for edge deployment. You will also examine potential use cases for your deployed people counter app.

### Course 2: Hardware for Computer Vision Deep Learning Application Deployment
Grow your expertise in choosing the right hardware. Identify key hardware specifications of various hardware types (CPU, VPU, FPGA, and Integrated GPU). Utilize the Intel® DevCloud for the Edge to test model performance and deploy power-efficient deep neural network inference on on the various hardware types. Finally, you will distribute workload on available compute devices in order to improve model performance.

**Lesson 1: Introduction to Hardware at the Edge.**
- Outcomes: 
  - Describe the importance of selecting the right hardware and the process involved in doing so.
- Exercises:
  - [Using Intel DevCloud][17]

**Lesson 2: CPU and Integrated GPU.**
- Outcomes: 
  - Identify the key specifications of Intel® CPUs and Integrated GPUs. 
  - Use the Intel® Devcloud for the Edge for running deep learning models on the CPU and Integrated GPU
- Exercises:
  - [CPU and the DevCloud][18]
  - [IGPU and the DevCloud][19]

**Lesson 3: Vision Processing Units.**
- Outcomes:
  - Identify the key specifications of Intel® VPUs
  - Use the Intel® DevCloud for the Edge for running deep learning
models on the VPU
  - Use the MULTI Plugin to get more consistent performance
- Exercises:
  - [VPU and the DevCloud][20]
  - [Walkthrough: Multi-Device Plugin and the DevCloud][21]
  - [Multi Device Plugin on DevCloud][22]

**Lesson 4: Field Programmable Gate Arrays.**
- Outcomes: 
  - Identify the key specifications of Intel® FPGAs
  - Use the Intel® DevCloud for the Edge for running deep learning
models on the FPGA
  - Use the HETERO Plugin to enable efficient hardware utilization
- Exercises:
  - [Walkthrough: FPGA and the DevCloud][23]
  - [FPGA and the DevCloud][24]
  - [Heterogeneous Plugin on DevCloud][25]

**Project: [Design a Smart Queuing System][26].**
- In this project, you will be given a real-world scenario of building a queuing system for three different clients in three different industry sectors. The sectors will consist of retail, manufacturing, and transportation. Each client will have their own set of constraints and requirements. You’ll use your knowledge of hardware specifications to identify which hardware types might work, and then you’ll test the application using the Intel® DevCloud for the Edge to see which hardware performs best. Finally, after reviewing your test results and considering the constraints and requirements of the client, you will propose a hardware solution and justify your selection.

### Course 3: Optimization Techniques and Tools for Computer Vision Deep Learning Applications
Learn how to optimize your model and application code to reduce inference time when running your model at the edge. Use different software optimization techniques to improve the inference time of your model. Calculate how computationally expensive your model is. Use DL Workbench to optimize your model and benchmark the performance of your model. Use a VTune amplifier to find and fix hotspots in your application code. Finally, package your application code and data so that it can be easily deployed to multiple devices.

**Lesson 1: Introduction to Software Optimization.**
- Outcomes:
Describe why Software Optimization is important
  - Identify the different fundamental optimization techniques 
  - Use different metrics to measure your model performance 
  - Identify when and how to use optimization techniques

**Lesson 2: Reducing Model Operations.**
- Outcomes:
  - Calculate the number of operations in a model
  - Implement optimization techniques that improve
performance by reducing the number of model operations
  - Use the Intel® Distribution of OpenVINOTM Toolkit to measure
the effects of different optimization techniques on the performance of your model
- Exercises:
  - [Pooling Performance][27]
  - [Separable Convolutions Performance][28]
  - [Measuring Layerwise Performance][29]

**Lesson 3: Reducing Model Size.**
- Outcomes:
  - Implement optimization techniques that improve performance by reducing the number of model operations
  - Use DL Workbench to quantize and measure the performance of your model

**Lesson 4: Other Software Optimization Techniques.**
- Outcomes: 
  - Use VTune Amplifier to measure hotspots in your application code
  - Package your application code and data
- Exercises:
  - [Packaging Your Application][30]
  - [Deploying Runtime Package][31]
  
**Project: [Build a Computer Pointer Controller][32].**
- In this project, you will use models available in the Intel® Distribution of OpenVINOTM Toolkit to control your computer pointer using your eye gaze. You will first have to identify faces and extract a face from an input video stream captured from a webcam or a video file. Then you will need to extract facial landmarks and also use a head pose estimation model to find the orientation of the extracted face. Using the head pose and facial landmarks, you will find the orientation of the eye gaze using a gaze estimation model. Finally, you will need to move the mouse pointer in the direction of the eye gaze. This project will demonstrate your ability to run multiple models in the same machine, and coordinate and optimize the flow of data between those models.

## License
This project is licensed under the MIT License. See [LICENSE][33] for details.

## Milestones
- 2020-05-28: Completed 3-course Nanodegree program.

[//]: # (Links Section)
[1]:https://github.com/robstraker/ai-for-iot-developers-udacity/BIO.md
[2]:https://www.udacity.com

[3]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-1-exercises/loading-pre-trained-models
[4]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-1-exercises/pre-processing-inputs
[5]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-1-exercises/deploy-an-app-at-the-edge
[6]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-2-exercises/convert-a-tf-model
[7]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-2-exercises/convert-a-caffe-model
[8]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-2-exercises/convert-an-onnx-model
[9]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-2-exercises/custom-layers
[10]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-3-exercises/feed-an-ir-to-the-inference-engine
[11]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-3-exercises/inference-requests
[12]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-3-exercises/integrate-into-an-app
[13]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-4-exercises/handling-input-streams
[14]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-4-exercises/process-model-outputs
[15]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/lesson-4-exercises/server-communications
[16]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-1-edge-ai-fundamentals-with-openvino/project-deploy-a-people-counter-app-at-the-edge

[17]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-1-exercises/using-intel-devcloud
[18]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-2-exercises/cpu-and-the-devcloud
[19]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-2-exercises/igpu-and-the-devcloud
[20]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-3-exercises/vpu-and-the-devcloud
[21]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-3-exercises/walkthrough-multi-device-plugin-and-the-devcloud
[22]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-3-exercises/multi-device-plugin-on-devcloud
[23]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-4-exercises/walkthrough-fpga-and-the-devcloud
[24]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-4-exercises/fpga-and-the-devcloud
[25]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/lesson-4-exercises/heterogeneous-plugin-on-devcloud
[26]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-2-hardware-for-computer-vision-deep-learning-application-deployment/project-design-a-smart-queuing-system

[27]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-3-optimization-techniques-and-tools-for-computer-vision-deep-learning-applications/lesson-2-exercises/pooling-performance
[28]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-3-optimization-techniques-and-tools-for-computer-vision-deep-learning-applications/lesson-2-exercises/separable-convolutions-performance
[29]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-3-optimization-techniques-and-tools-for-computer-vision-deep-learning-applications/lesson-2-exercises/measuring-layerwise-performance
[30]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-3-optimization-techniques-and-tools-for-computer-vision-deep-learning-applications/lesson-4-exercises/packaging-your-application
[31]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-3-optimization-techniques-and-tools-for-computer-vision-deep-learning-applications/lesson-4-exercises/deploying-runtime-package
[32]:https://github.com/robstraker/ai-for-iot-developers-udacity/course-3-optimization-techniques-and-tools-for-computer-vision-deep-learning-applications/project-computer-pointer-controller

[33]:https://github.com/robstraker/ai-for-iot-developers-udacity/LICENSE