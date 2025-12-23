\# Automated Yoga Pose Classification Using Deep Learning and MediaPipe Pose Estimation



\*\*CSC173 Intelligent Systems Final Project\*\*  

\*Mindanao State University - Iligan Institute of Technology\*  

\*\*Student:\*\* \[Kervin Lemuel D. Paalisbo ], \[2022-0076]  

\*\*Semester:\*\* \[e.g., AY 2024-2025 Sem 2]  



\[!\[Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org) \[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org) \[!\[MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)](https://google.github.io/mediapipe/)



---



\## Abstract



Yoga practice is growing rapidly in the Philippines, yet access to qualified instructors remains limited, particularly in rural areas of Mindanao. Many practitioners struggle to identify poses correctly, leading to ineffective practice or potential injuries. This project presents an efficient yoga pose classification system using MediaPipe pose estimation and a lightweight Multi-Layer Perceptron (MLP) classifier. Rather than processing raw pixel data, we extract 10 geometric features (arm angles, torso orientation, body spread) from detected body landmarks, significantly reducing computational requirements while maintaining high accuracy. The model was trained on 920 images across five fundamental yoga poses (Downward Dog, Goddess, Plank, Tree, Warrior 2) and achieved 92% validation accuracy and 89% test accuracy. Our feature-based approach enables real-time inference (~0.2s per image) suitable for mobile deployment, making yoga instruction more accessible. The system provides confidence scores for all poses, allowing practitioners to receive immediate feedback on their form. This work demonstrates that efficient pose classification can be achieved through intelligent feature engineering rather than computationally expensive end-to-end deep learning approaches.



---



\## Table of Contents



\- \[Introduction](#introduction)

\- \[Related Work](#related-work)

\- \[Methodology](#methodology)

\- \[Experiments \& Results](#experiments--results)

\- \[Discussion](#discussion)

\- \[Ethical Considerations](#ethical-considerations)

\- \[Conclusion](#conclusion)

\- \[Installation](#installation)

\- \[Usage](#usage)

\- \[References](#references)



---



\## Introduction



\### Problem Statement



Yoga has become increasingly popular worldwide as a holistic practice for physical and mental wellness. However, proper form is crucial to prevent injuries and maximize benefits. In the Philippines, particularly in Mindanao, access to certified yoga instructors is limited due to geographical constraints and cost barriers. Online tutorials lack real-time feedback mechanisms, leaving practitioners uncertain about their pose accuracy. This project addresses the need for an automated, accessible system that can identify yoga poses from images, providing instant feedback to practitioners regardless of their location or economic status.



\### Objectives



\- \*\*Objective 1:\*\* Achieve >90% validation accuracy and >88% test accuracy in classifying five fundamental yoga poses

\- \*\*Objective 2:\*\* Develop an efficient feature-based approach using MediaPipe pose estimation that reduces training time from hours to minutes per epoch

\- \*\*Objective 3:\*\* Create a practical inference system capable of real-time pose classification (<0.3s per image) suitable for mobile deployment

\- \*\*Objective 4:\*\* Implement comprehensive evaluation with confusion matrix analysis to identify pose similarities and classification patterns



!\[Yoga Pose Examples](images/yoga\_poses\_overview.png)



---



\## Related Work



\- \*\*MediaPipe Pose \[1]:\*\* Google's BlazePose provides real-time body pose tracking with 33 landmarks, achieving state-of-the-art accuracy on mobile devices. We leverage this for robust feature extraction.



\- \*\*Skeleton-Based Action Recognition \[2]:\*\* Research shows that geometric features from pose keypoints can effectively classify human poses while being computationally lighter than CNN approaches on raw images.



\- \*\*Yoga Pose Classification with CNNs \[3]:\*\* Prior work using ResNet and VGG architectures achieved 85-90% accuracy but required significant computational resources and large training datasets.





---



\## Methodology



\### Dataset



\- \*\*Source:\*\* Kaggle Yoga Pose Dataset (~1,550 images)

\- \*\*Classes:\*\* 5 yoga poses

&nbsp; - Downward Dog (downdog)

&nbsp; - Goddess (goddess)  

&nbsp; - Plank (plank)

&nbsp; - Tree (tree)

&nbsp; - Warrior 2 (warrior2)

\- \*\*Split:\*\* 60% train (920 images) / 10% validation (160 images) / 30% test (470 images)

\- \*\*Preprocessing:\*\* 

&nbsp; - RGB conversion

&nbsp; - MediaPipe pose landmark detection (33 keypoints)

&nbsp; - Feature extraction: 10 geometric features computed from landmarks

&nbsp; - Feature caching: All features pre-computed and saved to disk for efficient training



\### Architecture



!\[Model Architecture](images/architecture\_diagram.png)



\*\*Pipeline Overview:\*\*

```

Input Image → MediaPipe Pose Landmarker → 10 Geometric Features → MLP Classifier → Pose Prediction

```



\*\*Feature Engineering:\*\*

From the 33 body landmarks, we extract 10 discriminative features:

1\. Left arm angle (shoulder-to-elbow)

2\. Right arm angle (shoulder-to-elbow)

3\. Shoulder width (horizontal distance)

4\. Hip width (horizontal distance)

5\. Left torso angle (shoulder-to-hip)

6\. Right torso angle (shoulder-to-hip)

7\. Left leg angle (hip-to-knee)

8\. Right leg angle (hip-to-knee)

9\. Center of mass Y-coordinate

10\. Body spread (horizontal extent)



\*\*MLP Classifier:\*\*

\- \*\*Layer 1:\*\* Linear(10 → 64) + BatchNorm + ReLU + Dropout(0.4)

\- \*\*Layer 2:\*\* Linear(64 → 128) + BatchNorm + ReLU + Dropout(0.4)

\- \*\*Layer 3:\*\* Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)

\- \*\*Output:\*\* Linear(64 → 5) + Softmax



\*\*Hyperparameters:\*\*



| Parameter | Value |

|-----------|-------|

| Batch Size | 64 |

| Learning Rate | 1e-3 |

| Weight Decay | 1e-4 |

| Epochs | 30 |

| Optimizer | AdamW |

| LR Scheduler | ReduceLROnPlateau |

| Loss Function | Cross-Entropy |



\### Training Code Snippet



```python

\# Feature extraction (one-time preprocessing)

features\_dict = preprocess\_dataset(

&nbsp;   dataset\_dir='/content/drive/MyDrive/archive/DATASET',

&nbsp;   cache\_file='yoga\_features\_cache.pkl'

)



\# Model training

model = FeatureClassifier(input\_dim=10, num\_classes=5)

trainer = Trainer(model, train\_loader, val\_loader, device='cuda')

best\_acc = trainer.train(num\_epochs=30)

```



---



\## Experiments \& Results



\### Metrics



\*\*Per-Class Performance (Test Set):\*\*



| Pose | Precision | Recall | F1-Score | Support |

|------|-----------|--------|----------|---------|

| Downward Dog | 0.95 | 0.99 | 0.97 | 97|

| Goddess | 0.60 | 0.61 | 0.61 | 80|

| Plank | 0.85 | 0.98 | 0.91 | 115|

| Tree |0.87 | 0.96 | 0.91 | 69|

| Warrior 2 | 0.86 | 0.87 | 0.86 | 93 |



!\[Training Curves](visualization/training_history.png)



\*Figure 1: Training and validation loss curves shows the model is training well"



\### Confusion Matrix



!\[Confusion Matrix](visualization/confusion_matrix.png)



\*Figure 2: Confusion matrix showing Downward Dog pose has highest accuracy (99%), while Goddess and Warrior 2 are occasionally confused due to similar wide-stance positions\*



\### Demo



\[Video Demo: \[demo/CSC173\_Paalisbo\_Final.mp4]



---



\## Discussion



\### Strengths



\- \*\*Computational Efficiency:\*\* Feature-based approach reduces training time from 9+ minutes per epoch to ~5-10 seconds per epoch after initial feature extraction

\- \*\*Robust to Variations:\*\* MediaPipe's pose estimation handles body types and camera angles effectively

\- \*\*Computationally light:\*\* Low computational requirements make deployment on smartphones feasible

\- \*\*Interpretable Features:\*\* Geometric features provide clear understanding of what distinguishes different poses



\### Limitations



\- \*\*Limited Pose Library:\*\* Currently supports only 5 poses; expansion to 20+ poses would increase practical utility

\- \*\*Single-Person Detection:\*\* Cannot handle multiple practitioners in the same frame

\- \*\*Landmark Dependency:\*\* Performance degrades when body parts are occluded or out of frame

\- \*\*Static Analysis:\*\* Current system processes images only; video sequence analysis could capture pose transitions



\### Key Insights



\- \*\*Confusion Patterns:\*\* Wide-stance poses (Goddess, Warrior 2) show highest confusion, suggesting additional features like arm height could improve discrimination

\- \*\*Data Augmentation Alternative:\*\* Feature caching eliminates the traditional need for image augmentation, significantly speeding up the pipeline



---



\## Ethical Considerations



\### Bias and Fairness

\- \*\*Body Type Representation:\*\* Dataset may be skewed toward certain body types, potentially affecting accuracy for users with different physiques or abilities

\- \*\*Cultural Sensitivity:\*\* Yoga has sacred origins; the system should be positioned as a learning aid, not a replacement for traditional instruction

\- \*\*Accessibility:\*\* While the tool increases access, it may disadvantage practitioners in areas with poor internet connectivity needed for initial setup



\### Potential Misuse

\- \*\*Surveillance Concerns:\*\* Pose estimation technology could be repurposed for unauthorized monitoring

\- \*\*Medical Misrepresentation:\*\* System should not be marketed as a medical or physical therapy tool without proper validation

\- \*\*Misinformation Risk:\*\* Incorrect classifications could reinforce improper form if users blindly trust the system



\### Mitigation Strategies

\- Clearly document system limitations and intended use cases

\- Recommend users consult certified instructors for comprehensive training

\- Implement confidence thresholds to flag uncertain predictions

\- Make source code and methodology transparent for community review



---



\## Conclusion



This project successfully demonstrates an efficient approach to yoga pose classification using feature-based deep learning. By extracting geometric features from MediaPipe pose landmarks and training a lightweight MLP classifier, we achieved 76.9% test accuracy while maintaining real-time inference capabilities.



\### Key Achievements

\- Achieved good accuracy 

\- Demonstrated successful deployment on free-tier Google Colab hardware



\### Future Directions

1\. \*\*Expanded Pose Library:\*\* Incorporate 20-30 additional common yoga poses to increase practical utility

2\. \*\*Video Sequence Analysis:\*\* Extend to continuous pose tracking with temporal smoothing for more accurate real-world usage

3\. \*\*Mobile App Deployment:\*\* Port model to TensorFlow Lite for on-device inference on iOS/Android

4\. \*\*Pose Correction Feedback:\*\* Beyond classification, provide specific guidance on form improvement (e.g., "Raise left arm 15° higher")

5\. \*\*Multi-Person Support:\*\* Enable simultaneous tracking of multiple practitioners for group classes

6\. \*\*Accessibility Features:\*\* Add voice feedback for visually impaired users and support for modified poses for different ability levels



---



\## Installation





\### Google Colab Notebook



Open the complete training pipeline in Colab:



\[!\[Open In Colab](https://colab.research.google.com/drive/1se69AcwWDLyVGeqZFhFZJ54zKGLEQfLM?usp=sharing)



---



\## References



\[1] Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., \& Grundmann, M. (2020). BlazePose: On-device Real-time Body Pose tracking. \*arXiv preprint arXiv:2006.10204\*.



\[2] Verma, M., Kumawat, S., Nakashima, Y., \& Raman, S. (2020). Yoga-82: A New Dataset for Fine-grained Classification of Human Poses. \*IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops\*.



---

