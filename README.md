# Project Description

This project implements a sophisticated time series analysis framework leveraging a modified **Temporal Fusion Transformer (TFT)** architecture. It's designed to process and classify sequential data, extracting intricate patterns and relationships within complex temporal datasets.

## Core Functionality:

The project performs the following key functions:

1.  **Data Preprocessing and Feature Engineering:**
    *   Handles both numerical and categorical features within sequential inputs.
    *   Implements label encoding for categorical variables and standardization for numerical features.
    *   Engineers new features from existing ones to enhance model learning, such as calculated differences, logarithmic returns, and liquidity measures.

2.  **Sequence Preparation:**
    *   Organizes data into fixed-length sequences, suitable for input to the TFT model.
    *   Facilitates the creation of training, validation, and testing datasets.

3.  **Temporal Fusion Transformer (TFT) Model:**
    *   Employs a custom TFT architecture tailored for classification tasks.
    *   Integrates several advanced modules, including:
        *   **Variable Selection Network (VSN):** Dynamically selects relevant features at each time step.
        *   **Gated Residual Networks (GRNs):** Enhance gradient flow and enable learning of complex non-linear relationships.
        *   **Long Short-Term Memory (LSTM) Encoder:** Captures long-range dependencies within the sequences.
        *   **Multi-Head Attention:** Identifies and focuses on the most informative parts of the input sequences.
        *   **Positional Encoding:** Incorporates information about the order of elements within the sequences.
        * **TimeDistributed Layers:** Apply operations to each timestep individually.

4.  **Training and Validation:**
    *   Utilizes a standard training loop with optimization via stochastic gradient descent (e.g., AdamW).
    *   Implements a learning rate scheduler (ReduceLROnPlateau) to fine-tune the learning process.
    *   Employs early stopping to prevent overfitting based on validation performance.

5.  **Evaluation:**
    *   Provides metrics to assess the classification performance of the model, such as loss and accuracy.
    *   Saves the best performing model based on validation set performance.

6.  **Prediction:**
    *   Loads a saved model to perform classification on new, unseen sequences.
    *   Outputs the predicted class labels for each input sequence.
    *   Generates a submission file with the predicted class labels.

## Purpose:

This project serves as a powerful and flexible tool for **classifying sequential data**. It is capable of:

*   **Handling heterogeneous data:** Effectively processing a mixture of numerical and categorical inputs.
*   **Learning complex temporal patterns:** Capturing intricate relationships and dependencies within time series data.
*   **Adapting to different sequence lengths:** The model can be adjusted for various input sequence lengths.
*   **Providing accurate classifications:** Through its sophisticated architecture and training process.
*   **Feature extraction**: It extracts the most important features using techniques like VSN.
