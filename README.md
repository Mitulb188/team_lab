# Combining Prosodic and Acoustic Features for Audio Deepfake Detection

This repository contains the source code for a research project focused on the detection of audio deepfakes through the fusion of acoustic and prosodic features. The primary model employs a cross-attention mechanism to effectively integrate these distinct modalities.

## 📖 Overview

The emergence of hyper-realistic synthetic speech from advanced TTS and VC systems presents a growing threat. These "audio deepfakes" can be weaponized for fraud, impersonation, and propaganda, making robust detection a critical necessity. Our project addresses this challenge with a novel framework that moves beyond standard acoustic analysis. We incorporate prosodic features—the musicality of speech, including its rhythm and intonation—to create a more holistic analysis. By evaluating both the sound itself and the way it is spoken, our model can more accurately distinguish between genuine human speech and its synthetic counterparts.

## 🎯 Key Contributions

* **Novel Architecture Development:** A cross-attention Transformer model was successfully developed and implemented to fuse acoustic (CQCC) and prosodic feature sets for enhanced classification.
* **High-Performance Detection:** The proposed model demonstrates state-of-the-art performance, achieving a test Equal Error Rate (EER) of 6.68% on the standardized ASVspoof 2019 LA dataset.
* **Empirical Feature Analysis:** Ablation studies provided critical insights into feature importance, demonstrating that a judiciously selected set of six prosodic features, particularly Harmonics-to-Noise Ratio (HNR) and shimmer, can yield superior performance compared to a larger, more comprehensive set of 23 features.

## ⚙️ Methodology

The core of the proposed methodology is a deep learning model engineered to process two distinct feature streams extracted from an input audio signal.

1.  **Acoustic Feature Extraction:** Constant Q Cepstral Coefficients (CQCC) are utilized to capture the spectral texture and timbral characteristics of the audio.
2.  **Prosodic Feature Extraction:** Features corresponding to fundamental frequency (F0), jitter, shimmer, and Harmonics-to-Noise Ratio (HNR) are extracted using the Parselmouth and openSMILE toolkits to model the prosodic qualities of the speech.
3.  **Cross-Attention Fusion Mechanism:** A Transformer-based architecture facilitates the fusion of these modalities. This mechanism allows the prosodic features to dynamically query the acoustic feature representations, thereby enabling the model to focus on the most salient spectral information within a given prosodic context.

*Figure 1: The proposed cross-attention model architecture for fusing CQCC and prosodic features.*

## 🗂️ Repository Contents

* **`final_code.ipynb`**: A Jupyter Notebook containing the end-to-end implementation of the experimental pipeline, including data preprocessing, feature engineering, model training, and subsequent evaluation.
* **`TransformerFusion_PyTorch_Best_Top6feat.pth`**: A serialized PyTorch model checkpoint for our proposed architecture.

## 🛠️ Requirements

This project requires Python 3.x. The necessary libraries can be installed via pip:

```
pip install torch pandas numpy librosa parselmouth opensmile audiomentations
```

## 🚀 Execution Instructions

1.  Ensure all library dependencies listed in the Requirements section are installed.
2.  Confirm that the ASVspoof 2019 dataset is accessible at the path specified within the notebook.
3.  Open and execute the cells in **`Final_code.ipynb`** to run the complete data preprocessing, feature extraction, model training, and evaluation pipeline.
