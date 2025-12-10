\# CSC173 Deep Computer Vision Project Proposal

\*\*Student:\*\* \[Kervin Lemuel D. Paalisbo], \[2022-0076]  

\*\*Date:\*\* \[December 12,2025]



\## 1. Project Title 

\[Posture-based Scoliosis Screening using Deep Learning and Pose Estimation]



\## 2. Problem Statement

\[Scoliosis, an abnormal curvature of the spine, is commonly underdiagnosed in school-aged children in the Philippines due to limited access to specialized medical imaging. Traditional diagnosis relies on X-rays, which are costly and involve radiation exposure. This project aims to develop a computer vision system that detects potential scoliosis through human posture images which provides a low-cost and accessible early screening tool.

\## 3. Objectives

* Develop and train a deep learning model to detect scoliosis risk from posture images with high sensitivity.
* Extract human keypoints using pose estimation to quantify asymmetry in shoulders, hips, and spine.
* Implement a complete training pipeline including data preprocessing, augmentation, model training, validation, and evaluation
* Provide visual explanations (heatmaps, asymmetry lines) to improve interpretability and trust.



\## 4. Dataset Plan

\- Source: \[Scoliosis1K-Pose or relevant pose/back-photo dataset]

\- Classes: \[Healthy \& Good Posture, Scoliosis, Bad Posture]

\- Acquisition: Publicly available dataset download, supplemented with synthetic or manually collected images for “bad posture” class to balance training.



\## 5. Technical Approach

\- Architecture sketch

&nbsp;	- Input: RGB images of standing subjects or extracted keypoints

&nbsp;	- Pose Estimation: MediaPipe/OpenPose for 2D skeleton keypoints

&nbsp;	- Feature Extraction: Compute asymmetry metrics (shoulder tilt, hip tilt, spine deviation)

&nbsp;	- Classification/Regression: Simple CNN or MLP predicting class probabilities (healthy, scoliosis, bad posture) or scoliosis risk score

&nbsp;	- Output: Probability scores + visual overlay highlighting asymmetric regions



* Model: ResNet50 fine-tuned on extracted keypoint images or CNN/MLP on structured asymmetry features



* Framework: PyTorch



* Hardware: Google Colab with GPU

\## 6. Expected Challenges \& Mitigations

\- Challenge: Small dataset

\- Solution: Augmentation



* Challenge: Misclassification between scoliosis and poor posture
* Solution : Include dedicated "bad posture" class



