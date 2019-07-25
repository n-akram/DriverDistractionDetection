# DriverDistractionDetection
Driver distraction detection using StateFarm database and CNN model



######################### INFORMATION #################################


This repository cosists of a driver distraction detection using StateFarm database and CNN model. Two models have been trained with similar structure. One that classifies only mobile based distraction and another that classifies all distractions. The classes used are as follows.

    C0: safe driving
    C1: texting - right
    C2: talking on the phone - right
    C3: texting - left
    C4: talking on the phone - left
    C5: operating the radio
    C6: drinking
    C7: reaching behind
    C8: hair and makeup
    C9: talking to passenger

Classes used by model classifying only mobile based distractzions: C0 to C4

Implementation from the paper "A Deep Learning Approach to Detect Distracted Drivers Using a Mobile Phone"

DOI: https://doi.org/10.1007/978-3-319-68612-7_9 


Database used: StateFarm driver distraction detection.

https://www.kaggle.com/c/state-farm-distracted-driver-detection/data


#################### Technical INFORMATION ##############################

Implemented using: TensorKeras environment
Activate using : source TensorKeras/bin/activate


Environment details:
Python version: 3.5.2
Tensorflow backend : 1.14.0 (with GPU)
Keras : 2.2.4
Open CV : 4.1.0

System configuration:
OS: Linux
CPU: Intel core i9-9900K
GPU: GeForce RTX 2080 Ti

############################ To Run #####################################

1. Create virtual environment using: 
    virtualenv TensorKeras
    Virtualenv packge can be installed using : pip install virtualenv
    
2. Install the relevant packages from "requirements.txt"
    pip install -r requirements.txt

3. Activate the environment: 
    source TensorKeras/bin/activate

4. Run "python main.py" : for all classes predictions

5. Run "python main.py -m": for mobile only classes
