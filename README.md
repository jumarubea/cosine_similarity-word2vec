# Word2Vec-based Word Similarity and Country Prediction

This repository implements a solution for measuring word similarities using Word2Vec, alongside two similarity metrics: Euclidean Distance and Cosine Similarity. Additionally, it leverages cosine similarity to predict countries based on cities. The model is trained using the Google News Word2Vec dataset (300-dimensional vectors). The repository contains the following core components:

- **Main Script (`main.ipynb`)**: The main Jupyter notebook where you can visualize and interact with the Word2Vec model.
- **Model (`model.py`)**: Contains the logic for fetching country predictions using cosine similarity and embeddings.
- **Utility Functions (`utils.py`)**: Helper functions for preprocessing, data handling, and visualization.
- **PCA Visualization (`visualize_pca.py`)**: Visualizes high-dimensional word vectors in 2D using PCA.
- **Requirements File (`requirements.txt`)**: Lists the Python dependencies necessary for running the project.



## Installation

To run the project, ensure you have Python 3.6+ and then install the dependencies.

### Clone the repository:

```bash
git clone https://github.com/jumarubea/cosine_similarity-word2vec.git
cd cosine_similarity-word2vec
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Files Overview

### `main.ipynb`

This Jupyter notebook demonstrates how to:
- Load the Word2Vec embeddings.
- Calculate word similarities using Euclidean and Cosine Similarity.
- Predict a country from a city name using cosine similarity and word embeddings.

### `model.py`

Contains the core logic to:
- Load and utilize pre-trained Word2Vec embeddings.
- Predict the country for a given city name using cosine similarity.
- Calculate Euclidean and Cosine Similarities between words.

### `utils.py`

Utility functions for:
- Loading Word2Vec embeddings.
- Preprocessing and text cleaning.
- Other helper functions used in the project.

### `visualize_pca.py`

This script visualizes the high-dimensional word embeddings in 2D space using **PCA** (Principal Component Analysis).

### `requirements.txt`

Lists the Python packages required to run the project. Some key packages include:
- `gensim` for Word2Vec models
- `numpy` for numerical operations
- `matplotlib` for plotting
- `pandas` for handling data
- `nltk` for tokenization


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google News Word2Vec Embeddings**: Pre-trained Word2Vec embeddings from Google News dataset.
- **gensim**: For Word2Vec model handling and similarity calculations.
- **Coursera**
