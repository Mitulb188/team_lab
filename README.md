Combining Prosodic and Acoustic Features for Audio Deepfake Detection
This repository contains the source code for the research project on detecting audio deepfakes by fusing acoustic and prosodic features. The primary model uses a cross-attention mechanism to effectively combine these modalities.

📖 Overview
The rise of highly realistic Text-to-Speech (TTS) and Voice Conversion (VC) systems has created a new threat: audio deepfakes. These synthetic voices pose serious risks, from fraud to the spread of disinformation. This project develops a robust detection system that goes beyond traditional acoustic analysis by also considering prosody—the rhythm, stress, and intonation of speech. By analyzing both what is said (spectral content) and how it is said (prosodic delivery), our model learns a more comprehensive representation to distinguish between genuine and fake audio.

🎯 Project Achievements
State-of-the-Art Model: Developed a novel cross-attention Transformer model that successfully fuses acoustic (CQCC) and prosodic features.

High-Performance Detection: The proposed model achieved a test Equal Error Rate (EER) of 6.68% on the ASVspoof 2019 LA dataset.

Critical Feature Insights: Our ablation studies demonstrated that a small, well-chosen set of 6 prosodic features (with HNR and shimmer being most important) can outperform a much larger set of 23, highlighting the importance of careful feature selection.

⚙️ Methodology
The core of our approach is a deep learning model that processes two distinct feature streams from an audio input.

Acoustic Features: We use Constant Q Cepstral Coefficients (CQCC) to capture the spectral texture and timbre of the audio.

Prosodic Features: We extract features like fundamental frequency (F0), jitter, shimmer, and Harmonics-to-Noise Ratio (HNR) using Parselmouth and openSMILE to model the speech's delivery.

Cross-Attention Fusion: A Transformer-based architecture uses the prosodic features to query the acoustic feature representations, allowing the model to focus on the most relevant spectral information in a given prosodic context.

Figure 1: The proposed cross-attention model architecture for fusing CQCC and prosodic features.

🗂️ Repository Structure
Final_code.ipynb: The main Jupyter Notebook containing the complete pipeline, from data preprocessing and feature engineering to model training and evaluation.

best_simple_ffnn_static_spoofing_detector.pth: A saved PyTorch model checkpoint from one of the baseline experiments.

Team_Lab_Paper.pdf: The research paper detailing the project's methodology, experiments, and findings.

🛠️ Requirements
This project requires Python 3.x. You can install the necessary libraries using pip:

pip install torch pandas numpy librosa parselmouth opensmile audiomentations

🚀 How to Run
Ensure you have all the required libraries installed.

Make sure the ASVspoof 2019 dataset is accessible at the path specified within the notebook.

Open and run the cells in Final_code.ipynb to execute the data preprocessing, feature extraction, model training, and evaluation pipeline.
