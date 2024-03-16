# Eye Gaze Detection Project

This project aims to develop a deep learning model for eye gaze detection using the MPII Gaze Detection dataset. The project involves pre-processing the dataset, training a deep learning model, and visualizing the results.

## Dataset

The MPII Gaze Detection dataset consists of images and corresponding gaze directions captured from various subjects. It contains both indoor and outdoor scenes with different head poses and lighting conditions.

Dataset Link: [MPII Gaze Detection Dataset]([](https://paperswithcode.com/dataset/mpiigaze)

## Pre-processing

The pre-processing steps involve:

1. Data loading: Load the dataset and extract relevant features such as eye images and corresponding gaze directions.
2. Data augmentation: Augment the dataset to increase its diversity and improve model generalization.
3. Data normalization: Normalize the input features to ensure numerical stability during training.

## Model Training

For model training, we will use a deep learning architecture tailored for eye gaze detection. The architecture will consist of convolutional neural networks (CNNs) followed by fully connected layers. The training process involves:

1. Splitting the dataset into training, validation, and test sets.
2. Training the model using a suitable optimization algorithm such as Adam or RMSprop.
3. Evaluating the model performance on the validation set and fine-tuning hyperparameters if necessary.
4. Testing the final model on the test set to assess its generalization ability.

## Visualization

After training the model, we will visualize its performance using various techniques such as:

1. Plotting training and validation curves to analyze model convergence and overfitting.
2. Visualizing the learned features using techniques like t-SNE or PCA.
3. Visualizing gaze predictions overlaid on input images to understand where the model focuses its attention.

## Usage

To replicate the experiment, follow these steps:

1. Download the MPII Gaze Detection dataset from the provided link.
2. Pre-process the dataset using the scripts provided in the `preprocessing/` directory.
3. Train the deep learning model using the scripts in the `training/` directory.
4. Visualize the results using the scripts in the `visualization/` directory.

## Dependencies

- Python 3.x
- TensorFlow or PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Contributors

- [Your Name](link_to_github_profile)
- [Collaborator's Name](link_to_github_profile)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

