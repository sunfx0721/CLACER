# CLACER: A Deep Learning Based Compilation Error Classification Method for Novice Programs

We propose a new category of compilation error based on the program tokens, which is the smallest unit of the program. Then we develop a neural network model CLACER (ClAssification of Compilation ERrors) based on TextCNN. CLACER performs better on extracting semantic features and statistical features from compiler error messages. 

To verify the effectiveness of our proposed category and corresponding method CLACER, we choose 16,926 student programs as our experimental subjects. Final experimental results indicate that our proposed classification category covers 16.5% more programs than the TEGCER category, the state-of-the-art category. Moreover, CLACER improves the compiler's localization effectiveness and with a 4.25% improvement on the TEGCER category. Further analysis shows that CLACER has a promising prediction performance for different error classes, and TextCNN is more suitable for constructing the compilation error classification model.


# Dataset

In this study, we construct the student code repositories from two publicly available datasets (i.e., The DeepFix dataset(http://iisc-seal.net/deepfix) and The TEGCER dataset(https://github.com/umairzahmed/tegcer)). These datasets are all curated from Introductory to C Programming (CS1) of college students.  
These assignments were recorded using a custom web-browser based IDE Prutor. 

The DeepFix datasets contains 6,971 programs spanning 93 assignments that fail to compile, each lengths range from 40 to 100 token. 

The TEGCER dataset contains 23,275 buggy programs with corresponding repairs. TEGCER dataset spans 40+ different programming assignments completed by  400+ first-year undergraduate students.

We  only use programs with single-line buggy statement.Finally, we select 2,911 programs  from 6,971 programs in the DeepFix dataset and 14,015 programs from 23,275 programs in the TEGCER dataset.


# Code structure

Our code structure is mainly composed of four parts

1. Generate the codedata class. This class is used to store the properties needed to process code.

    DataSetGenerator.py

2. Training model

    model_train.py

3. Forecast results

    model_predict.py

4. Analysis results

    result_analysis.py


# How to run code
Directly configure the parameters you need in the main.py and click Run.

