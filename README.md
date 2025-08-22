# MNIST Digit Classification Streamlit App

This project is a web app for classifying handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset with 99% accuracy. The app is built with Streamlit and allows users to:

- View and predict random MNIST test samples
- Upload their own digit images for prediction

<img width="1634" height="867" alt="image" src="https://github.com/user-attachments/assets/9ab8a6c2-d4a4-4b53-a0ba-ab3cb69e705a" />


## Getting Started

### 1. Clone the repository

```
git clone <your-repo-url>
cd Digit Image Classification
```

### 2. Set up a virtual environment (optional but recommended)

```
python -m venv my_venv
# Activate the environment:
# On Windows:
my_venv\Scripts\activate
# On Mac/Linux:
source my_venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```
streamlit run app.py
```

## File Structure

- `app.py` : Streamlit web app
- `requirements.txt` : Python dependencies
- `Model/mnist_cnn_model.h5` : Trained CNN model
- `Notebook/mnist.ipynb` : Jupyter notebook for model training
- `test_images/` : Example digit images for testing

## Notes

- Make sure `Model/mnist_cnn_model.h5` exists before running the app.
- For best results, uploaded images should be 28x28 pixels and grayscale.

## License

MIT
