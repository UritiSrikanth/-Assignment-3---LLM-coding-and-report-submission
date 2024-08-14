# -Assignment-3---LLM-coding-and-report-submission
LLM coding and report submission
**Overview**
This Jupyter Notebook demonstrates the process of training and evaluating a BERT-based model for sequence classification using a custom dataset. The objective is to classify different types of prompts into categories such as "command," "statement," or "question."
**Dataset**
The dataset used in this assignment is the "fka/awesome-chatgpt-prompts" dataset, which contains a collection of various prompts used for generating responses from AI models. The dataset is loaded using the Hugging Face datasets library.
**Dataset Structure**
•	Prompt: The main text input that we aim to classify.
•	Labels: Synthetic labels generated for the purpose of binary classification in this assignment.
**Model: BERT (Bidirectional Encoder Representations from Transformers)**
The model used in this assignment is the BERT model, specifically bert-base-uncased, which is pre-trained on a large corpus of English text. BERT is a transformer-based model that excels at understanding the context of words in a sentence, making it highly effective for natural language processing tasks such as sequence classification.
Tokenization
•	Tokenizer: We used the BertTokenizer from the Hugging Face Transformers library to tokenize the text data. The tokenizer converts the text into a format that the BERT model can understand (i.e., into token IDs).
•	Padding and Truncation: The text data is padded and truncated to a maximum length to ensure uniformity across the dataset.
**Training and Evaluation
Data Preparation**
•	Labelling: Since the dataset does not come with labels for classification, we generate synthetic binary labels for training purposes. The labels are randomly assigned as either 0 or 1.
•	Train-Test Split: The tokenized dataset is split into training (90%) and evaluation (10%) sets.
Model Configuration: 
•	Model: BertForSequenceClassification is used, configured for binary classification with two labels.
•	Training Arguments:
o	learning_rate: 2e-5
o	num_train_epochs: 3
o	per_device_train_batch_size: 16
o	per_device_eval_batch_size: 16
o	evaluation_strategy: "epoch"
o	logging_dir: './logs'
o	output_dir: './results'
o	save_total_limit: 1 (to limit the number of saved models)
**Training Process**
The model is trained using the Trainer API from the Hugging Face Transformers library. This simplifies the training loop and handles the training, evaluation, and logging processes.
**Evaluation Metrics**
•	Accuracy: The primary metric used to evaluate the model’s performance. The metric is computed using the evaluate library, specifically the accuracy metric.
•	Custom Metrics Callback: A custom callback is implemented to log training and evaluation metrics, including loss and accuracy after each epoch.
**Visualization**
•	Training and Validation Loss: The notebook plots the training and validation loss over the steps to visualize the model's learning progress.
•	Training and Validation Accuracy: Similarly, accuracy metrics are plotted to compare the model’s performance on the training and validation datasets.
**Inference**
After training, the model is tested on a set of provided sentences to classify them into "command," "statement," or "question." The true labels are compared with the predicted labels to evaluate the model's effectiveness.
**Conclusion**
This notebook illustrates the application of a BERT-based model for sequence classification. By leveraging the power of BERT, the model is trained to understand and classify different types of prompts, showing how transfer learning can be applied to specific tasks with minimal data.
