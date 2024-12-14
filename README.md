# MinML: Minimal GRU vs. Traditional GRU on Real-World Data

This repository contains the implementation of **MinML**, a project designed to evaluate and compare the performance of **Minimal GRU (MinGRU)** and **Traditional GRU** models on real-world sequential data. The analysis uses whale dive datasets, with steps for preprocessing, model training, evaluation, and result visualization.

The walkthrough is provided as a **Jupyter Notebook (\`.ipynb\`)** that can be run directly on **Google Colab**.

---

## About MinGRU vs. Traditional GRU: A Comparison for Bio-Telemetry Research

Behavioral analysis in bio-telemetry relies on interpreting complex time-series data such as animal movement, diving behaviors, and environmental interactions. Traditional GRUs are well-known for their ability to handle sequential data, but their complexity and resource-intensive training make them less ideal for large-scale or real-time behavioral studies.

MinGRU offers a lightweight alternative with fewer parameters, reducing computational load without compromising predictive accuracy. By eliminating unnecessary gates and focusing on essential temporal features, MinGRU is a better fit for bio-telemetry research where:

1. **Efficiency Matters**: MinGRU is more efficient for real-time applications, enabling researchers to process large datasets quickly.
2. **Low Resource Environments**: Field researchers often lack access to high-end GPUs; MinGRU provides a viable solution for edge devices or resource-constrained settings.
3. **Scalability**: With its reduced complexity, MinGRU scales effectively to larger datasets or extended observational periods.
4. **Behavior State Identification**: MinGRU excels in detecting behavioral patterns across time-series data, making it an invaluable tool for ecological and ethological studies.

This repository includes the first documented real-world comparative study of MinGRU and Traditional GRU models, providing detailed evaluations of accuracy, memory usage, and computational efficiency on annotated whale dive datasets.

---

## Project Highlights

- **Google Colab Integration**: Run the entire project without local setup—just open the notebook on Colab.
- **MinGRU Implementation**: A novel minimal architecture benchmarked against a Traditional GRU.
- **Real-World Data**: Uses whale dive datasets and annotations, demonstrating practical behavior modeling.
- **Comparative Analysis**: Detailed comparison of computational efficiency and model accuracy between MinGRU and Traditional GRU.
- **Interactive Notebook**: Step-by-step execution for hands-on exploration and visualization.

---

## The Walkthrough Notebook

The Jupyter Notebook in this repository, **\`MinML_Walkthrough.ipynb\`**, provides a guided exploration of the MinML pipeline. It is designed to run directly on Google Colab and includes the following steps:

1. **Data Preparation**:
   - Imports and preprocesses datasets, ensuring compatibility with PyTorch models.
   - Handles annotations and aligns them with datasets for behavior state labeling.
   
2. **Model Training**:
   - Implements and trains both MinGRU and Traditional GRU architectures.
   - Tracks performance metrics during training for comparison.

3. **Evaluation**:
   - Uses the evaluation module to compute metrics like accuracy and precision.
   - Visualizes confusion matrices and other diagnostic plots.

4. **Inference**:
   - Demonstrates how both models perform on unseen data.
   - Provides insights into behavioral state transitions based on predictions.

5. **Visualization**:
   - Offers tools for visualizing state predictions, data distributions, and performance comparisons.

The walkthrough ensures a comprehensive understanding of both models and their application to bio-telemetry datasets.

---

## Repository Contents

### File Overview

#### `data_processing.py`

This script is at the core of preparing the datasets for machine learning workflows. It includes functionality for loading raw datasets, handling missing values, and engineering features critical to the modeling pipeline. Specific features include depth velocity, pitch rate, and roll rate, which provide essential context for interpreting whale dive behaviors. 

Additionally, the script automates the labeling of behavioral states by aligning dataset samples with annotations, which are later used to train classification models. The dataset is split into training, validation, and testing subsets, ensuring compatibility with PyTorch's `Dataset` class. 

A significant highlight is its modular structure: individual preprocessing steps can be reused or extended for new datasets. It seamlessly handles both raw and pre-processed data formats, enabling flexibility in exploration and reproducibility in research workflows.

#### `evaluate_model.py`

This script provides robust tools for evaluating machine learning models. It calculates key metrics such as accuracy, precision, recall, and F1-score, giving researchers comprehensive insights into model performance. The script also includes a confusion matrix visualization module, offering a detailed view of how well the model predicts each class.

Built for scalability, it supports evaluating models across different datasets and configurations, making it suitable for comparative studies like the MinGRU vs. Traditional GRU analysis. Its ability to integrate with the visualization utilities enhances the interpretability of results for researchers and stakeholders.

#### `gru_models.py`

This script defines the architectures of the Minimal GRU (MinGRU) and Traditional GRU models, both implemented in PyTorch. MinGRU introduces a lightweight design with fewer parameters, focusing on computational efficiency without sacrificing accuracy. Traditional GRU follows the standard gated recurrent unit (GRU) architecture with support for multiple layers, catering to more complex modeling tasks.

The models are designed for classification tasks, with flexibility in input size, hidden dimensions, and output classes. Each model's forward pass processes time-series data to capture temporal dependencies, outputting predictions based on learned representations of sequential patterns. The script also includes utility functions for model initialization and parameter optimization.

#### `hmm_refinement.py`

This script applies Hidden Markov Models (HMMs) to refine behavioral state predictions. Using features like depth velocity and pitch rate, it models the sequence of observed states with probabilistic transitions between them. The HMM framework improves classification accuracy by leveraging temporal consistency and context.

One notable feature is its ability to reclassify ambiguous states using emission probabilities and transition matrices, effectively correcting misclassifications. This makes it a valuable tool for behavioral studies where temporal structure is as critical as individual observations.

#### `rjmcmc_module.py`

This script implements Reversible Jump Markov Chain Monte Carlo (RJMCMC), a Bayesian approach for discovering latent states in sequential data. It dynamically adjusts the number of states during inference, allowing for flexible modeling of complex behaviors.

By proposing state transitions and evaluating their posterior probabilities, it provides a mechanism for uncovering hidden patterns not immediately apparent in the data. This approach complements the HMM refinement, offering a deeper understanding of latent structures and variability in behavior.

#### `state_management_toolkit.py`

Combining HMM and RJMCMC methods, this script serves as a toolkit for managing behavioral states. It integrates probabilistic refinements and state discovery into a unified framework, simplifying exploratory and confirmatory analyses. 

The script offers utilities for preprocessing state sequences, validating transitions, and generating detailed reports on state distributions. Its modular design ensures compatibility with datasets of varying complexity and granularity.

#### `train_model.py`

This script manages the training pipeline for both MinGRU and Traditional GRU models. It supports hyperparameter tuning, loss tracking, and real-time metrics reporting during training. Designed with flexibility in mind, it integrates seamlessly with PyTorch's `DataLoader` for batch processing and supports GPU acceleration for large-scale training.

Key features include saving trained models, logging performance metrics, and handling interruptions gracefully. The script is optimized for iterative experimentation, making it easy to evaluate different configurations and datasets.

#### `transfer_learning.py`

This script enables transfer learning, allowing models trained on one dataset to adapt to new tasks or domains. By freezing certain layers and fine-tuning others, it accelerates training on small or related datasets. This is particularly useful in scenarios where labeled data is limited.

#### `visualization_utils.py`

This script provides a suite of tools for visualizing data, model outputs, and evaluation results. It supports generating state distribution plots, confusion matrices, and feature correlation heatmaps. Built on Matplotlib and Seaborn, it produces publication-quality visuals with customization options for aesthetics and clarity.

#### `whale_orchestrator.py`

Serving as the central pipeline manager, this script orchestrates the entire workflow from data preprocessing to model evaluation. It ties together the functionalities of the other scripts, enabling end-to-end experimentation with minimal manual intervention.

---

## License

This project is licensed under the MIT License. See the \`LICENSE\` file for more details.

---

Feel free to explore the notebook on Colab, and reach out via the repository’s Issues section with any questions or feedback.
