# Theia
State Of The Art Computer Vision Exam Proctoring System , Made using YoloV5 and Dlib 

<details>
<summary>Features</summary>
   
This software is designed to uphold the integrity of exams conducted on computer screens by detecting and preventing suspicious activities committed by candidates. It relies on the candidate's webcam to monitor and analyze their behavior during the exam.

### Key Features:

1. Violation Counter: This system maintains a violation counter that allows a predefined number of infractions within an adjustable limit. Once this limit is reached, an alert is immediately sent to the administrator, and the candidate's computer is rendered inoperable until further action is taken.

2. Violations Detection: The system actively monitors the candidate's actions and records violations with corresponding image evidence. These violations include:

   a. Using a mobile phone during the exam.
   
   b. Diverting attention away from the screen, such as peeking at others' screens or consulting external materials.
   
   c. Detecting multiple individuals in the webcam frame, signaling potential collaboration or unauthorized presence.
   
   d. Not detecting any person in the frame for an extended period, which may indicate candidate absence or an issue.
   
   e. (Upcoming Feature) Facial Recognition: This upcoming feature will ensure that only the designated candidate uses the computer for the exam, preventing unauthorized access.

In summary, our state-of-the-art computer vision-based exam proctoring system provides a robust solution to maintain exam integrity by monitoring and flagging suspicious activities, all while offering customizable violation tolerance levels and advanced features for enhanced security.
</details>
<details>
<summary>Installation</summary>
This software is designed to operate efficiently without the need for a GPU, relying solely on CPU resources. Therefore, it is essential to ensure that your system is equipped with a capable CPU.

### To get started with the installation process, follow these steps:

1. Begin by installing Python 3, which can be downloaded from the official Python website at https://www.python.org

2. Next, either clone the repository or download it as a ZIP file to your local machine.

3. Once you have the software's repository on your machine, navigate to the cloned folder. Open a terminal window in this folder and execute the following command to install the required dependencies. Please note that this step may take some time to complete:

```bash
pip install -r requirements.txt
```

4. Once the dependencies are successfully installed, you can initialize the application by running the following command in the terminal:

```bash
python3 webcam.py
```

These steps will set up and launch the software, allowing you to use it for monitoring and proctoring exams using your CPU resources.

</details>

