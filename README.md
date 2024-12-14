# MinML: Minimal GRU vs. Traditional GRU on Real-World Data

This repository contains the implementation of **MinML**, a project designed to evaluate and compare the performance of **Minimal GRU (MinGRU)** and **Traditional GRU** models on real-world sequential data. The analysis uses whale dive datasets, with steps for preprocessing, model training, evaluation, and result visualization.

The walkthrough is provided as a **Jupyter Notebook (`.ipynb`)** that can be run directly on **Google Colab**.

---

## Project Highlights

- **Google Colab Integration**: Run the entire project without local setup—just open the notebook on Colab.
- **MinGRU Implementation**: A novel minimal architecture benchmarked against a Traditional GRU.
- **Real-World Data**: Uses whale dive datasets and annotations, demonstrating practical behavior modeling.
- **Comparative Analysis**: Detailed comparison of computational efficiency and model accuracy between MinGRU and Traditional GRU.
- **Interactive Notebook**: Step-by-step execution for hands-on exploration and visualization.

---

## How to Run the Notebook

1. **Open on Google Colab**:
   - Navigate to the repository: [MinML GitHub Repository](https://github.com/GilRaitses/minML/).
   - Open `MinML_Walkthrough.ipynb` in Google Colab by clicking "Open in Colab" or uploading the file directly to Colab.

2. **Set Up the Environment**:
   - Install the required Python libraries using the commands in the notebook.
   - Follow the notebook’s steps to initialize the database and preprocess the datasets.

3. **Run the Pipeline**:
   - Train and evaluate both the **MinGRU** and **Traditional GRU** models on the datasets provided.
   - Compare their performance in terms of accuracy, efficiency, and memory usage.

4. **Visualize Results**:
   - Explore the provided visualizations comparing predictions against ground truth.
   - Analyze the comparative metrics for both models.

---

## Results Summary

This project provides a comparative analysis of **MinGRU** and **Traditional GRU**:

- **Efficiency**: MinGRU achieves comparable accuracy with reduced computational complexity, making it ideal for real-time applications.
- **Accuracy**: Both models are evaluated using annotated whale dive data, showcasing behavior predictions aligned with ground truth.
- **Significance**: To our knowledge, this is the first documented real-world comparison of MinGRU and Traditional GRU architectures.

---

## Next Steps

- Expand MinGRU applications to other sequential data challenges, such as EEG or motion tracking.
- Introduce additional lightweight architectures like Minimal LSTM.
- Optimize for faster training and inference on large-scale datasets.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to explore the notebook on Colab, and reach out via the repository’s Issues section with any questions or feedback.
