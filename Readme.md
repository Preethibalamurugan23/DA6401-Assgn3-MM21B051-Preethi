# DA6401 - MM21B051 - Preethi
## Sequence-to-Sequence Model for Character-Level Translation


This project implements a sequence-to-sequence (seq2seq) model for character-level translation. The model is designed to translate romanized text into a native language (Hindi in this case), using character-level tokenization.  It includes both a basic seq2seq model and an attention-based variant.  The data used is the Hindi language data from the Dakshina-transliteration dataset.

## Features

* **Seq2Seq Architecture:** The core of the model is an encoder-decoder architecture, which is well-suited for sequence translation tasks.
* **Character-Level Tokenization:** Both the input (romanized text) and the output (native language) are processed at the character level. This allows the model to handle a large vocabulary and capture fine-grained details in the text.
* **LSTM/GRU/RNN Support:** The encoder and decoder can be configured to use either LSTM, GRU, or simple RNN cells.
* **Attention Mechanism:** An attention mechanism is used to improve the model's performance by allowing the decoder to focus on the most relevant parts of the input sequence.
* **Dropout:** Dropout is applied to both the encoder and decoder to prevent overfitting.
* **Train and Prediction:** The code includes functions for training the model and generating translations for new input sequences.
* **NaN Handling:** The code includes a mechanism to handle NaN values in the input data.
* **Wandb Integration**: The code uses Weights & Biases (Wandb) for experiment tracking, hyperparameter tuning, and visualization.
* **Dakshina Dataset**: The code uses the Hindi language data from the Dakshina-transliteration dataset.

## Code Description

The project is organized as follows:

1.  **Loading and Preprocessing Data**: The code loads the training, development, and test datasets from the Dakshina dataset, and preprocesses the text data. This includes handling any missing values, converting text to lowercase, and adding start and end tokens.
2.  **Defining Model Architecture (Simple Seq2Seq)**:  The code defines the encoder-decoder architecture for a basic sequence-to-sequence model.
3.  **Wandb Hyperparameter Sweep**:  The code performs a hyperparameter sweep using Wandb to find the best configuration for the simple seq2seq model.
4.  **Training and Testing Best Model (Simple Seq2Seq) with Visualizations**: The best performing model from the hyperparameter sweep is trained, and the results are visualized using Wandb.
5.  **Defining Model Architecture (Attention-Based Seq2Seq)**: The code defines the encoder-decoder architecture with an attention mechanism.
6.  **Wandb Hyperparameter Sweep**: The code performs a hyperparameter sweep using Wandb to find the best configuration for the attention-based seq2seq model.
7.  **Training and Testing Best Model (Attention-Based Seq2Seq) with Visualizations**: The best performing model from the attention-based hyperparameter sweep is trained, and the results are visualized using Wandb.
8.  **Requirements**: A `requirements.txt` file is included, listing the Python dependencies for the project.

The main parts of the code are:

* **Import Libraries**: Imports necessary libraries, including TensorFlow, Keras, and pandas.
* **ReshapeInitialState Layer**: Defines a custom Keras layer to reshape the initial state of the decoder.
* **`build_and_train_model` Function**:
    * Builds the encoder and decoder models (both simple and attention-based).
    * Supports LSTM, GRU, and RNN cells.
    * Implements the attention mechanism (for the attention-based model).
    * Compiles and trains the model using the specified optimizer and loss function.
* **`translate_sentence` Function**:
    * Translates an input sequence using the trained encoder and decoder models.
    * Handles the decoder state and generates the output sequence character by character.
* **Main Execution Block**:
    * Defines the model configuration.
    * Loads and preprocesses the training, development, and test data.
    * Tokenizes the input and target texts at the character level.
    * Pads the sequences to the maximum length.
    * Builds and trains the model.
    * Prepares the encoder and decoder models for prediction.
    * Generates predictions on the test set.
    * Prints sample predictions and saves them to files.

## How to Use

1.  **Install Dependencies:**
    * Install the required Python libraries. It is recommended to create a virtual environment.
    * Use `pip install -r requirements.txt` to install all the necessary packages.

2.  **Prepare Data:**
    * The code uses the Hindi language data from the Dakshina-transliteration dataset.
    * Ensure that the lexicon files from the Dakshina dataset are stored in the appropriate directory. You may need to adjust the file paths in the code to point to the correct location of these files.
    * The code expects the data to be in a pandas DataFrame format, and will handle the loading.

3.  **Configure Wandb:**
    * Obtain your Wandb API key from your account settings.
    * Replace `YOUR_WANDB_API_KEY` in the code with your actual Wandb API key.

4.  **Configure Parameters:**
    * Adjust the parameters in the `best_params` dictionary in the `if __name__ == '__main__':` block to match your data and requirements.
    * Key parameters include:
        * `batch_size`: The batch size for training.
        * `cell_type`: The type of RNN cell to use ('lstm', 'gru', or 'simple_rnn').
        * `decoder_units`: The number of units in the decoder RNN.
        * `dropout_rate`: The dropout rate.
        * `embedding_dim`: The dimensionality of the embedding vectors.
        * `encoder_units`: The number of units in the encoder RNN.
        * `learning_rate`: The learning rate for the optimizer.
        * `optimizer`: The optimizer to use ('adam' or 'rmsprop').


5.  **Run the Code:** Run the Python script. The model will be trained, and predictions will be generated on the test set. The training process and results will be logged to your Wandb account.

## Input Data Format

The code uses the Hindi language data from the Dakshina-transliteration dataset. The data is expected to be in a pandas DataFrame with two columns:

* `romanized`: Contains the input text in romanized form.
* `native`: Contains the target text in the native language. This column *must* have `` and `` tokens.

For example:

| romanized   | native             |
| :---------- | :----------------- |
| hello       |  नमस्ते   |
| world       |  दुनिया   |


## Output

The code will print sample predictions on the test data and save all predictions to text files in a folder named "predictions". Training and validation metrics, as well as model performance, will be logged and visualized on Wandb.

## Model Configuration

The following parameters can be configured in the `best_params` dictionary:

* `batch_size`: Batch size used for training.
* `cell_type`: Type of RNN cell ('lstm', 'gru', or 'simple_rnn').
* `decoder_units`: Number of units in the decoder RNN.
* `dropout_rate`: Dropout rate.
* `embedding_dim`: Dimensionality of the embedding layer.
* `encoder_units`: Number of units in the encoder RNN.
* `learning_rate`: Learning rate for the optimizer.
* `optimizer`: Optimizer to use ('adam' or 'rmsprop').

## Preprocessing

The code performs the following preprocessing steps:

* Converts text to lowercase.
* Adds `` and `` tokens to the target sequences.
* Character-level tokenization.
* Padding sequences to the maximum length.

## Tokenization

The code uses Keras' `Tokenizer` to convert the text into sequences of integers, representing character indices. The tokenizer is fit on the training data for both input and output languages.

## Padding

The input and output sequences are padded to the maximum length using Keras' `pad_sequences`. This ensures that all sequences in a batch have the same length.

## Model Architecture

The model architecture is as follows:

* Encoder:
    * Input embedding layer.
    * LSTM/GRU/RNN layer(s).
    * Optionally, a dropout layer.
* Decoder:
    * Input embedding layer.
    * LSTM/GRU/RNN layer(s) with initial states from the encoder.
    * Optionally, a dropout layer.
    * Attention mechanism to attend over encoder outputs (for attention-based model).
    * Dense layer to predict the next character.

## Prediction

The `translate_sentence` function generates the translated sequence character by character. It uses the encoder to get the initial decoder state and the encoder outputs. Then, it iteratively predicts the next character using the decoder, updating the decoder state at each step. The prediction stops when the `` token is generated or the maximum target length is reached.

## Error Handling

The code handles `NaN` values in the input data during preprocessing.
.
