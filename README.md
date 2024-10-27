# Generative AI


**<u>Generative AI:</u>**

Generative AI is a type of AI that can create new content, such as text, images, music, audio, and videos. It's powered by large AI models, called foundation models, that can perform a variety of tasks, including summarization, Q&A, and classification.

---

**<u>Key Concepts in Language Model Interaction:</u>**
- **Prompt:** The input text provided to the model.
- **Inference:** The process of generating text based on the prompt.
- **Completion:** The output text produced by the model.

---

**<u>Context Window:</u>**

The context window refers to the total amount of text that the model can consider at one time. It is a critical factor since it limits the amount of information that can be used for generating responses.

---

## Prompt Engineering:
Prompt engineering is a critical aspect of working with language models (LMs), such as GPT (Generative Pre-trained Transformer). It involves crafting the input text (the prompt) in a way that effectively guides the Language Models towards generating the desired output (the completion). This process can sometimes require multiple iterations of refinement to achieve the best results, a practice known as prompt engineering.


**<u>In-context learning:**</u>

In-context learning is a powerful technique where examples or additional data are included within the prompt to help the model understand and perform the task better. It can significantly enhance the model’s ability to generate appropriate and accurate completions.


**<u>Zero-Shot vs. One-Shot vs. Few-Shot Inference:**</u>

- **Zero-shot inference:** Providing the model with no specific examples, just instructions.
- **One-shot inference:** Including a single example within the prompt to guide the model.
- **Few-shot inference:** Incorporating multiple examples to better demonstrate the desired output.

  ![alt text](image-1.png)

**<u>Effectiveness Across Model Sizes:**</u>

- **Larger models:** Typically excel at zero-shot inference, understanding tasks with little to no examples.
- **Smaller models:** May struggle with zero-shot inference but improve significantly with one-shot or few-shot examples.  


**<u>Limitations and Fine-Tuning:**</u>

While in-context learning is powerful, it’s important to ***remember the context window limitation.*** If including multiple examples does not improve model performance, fine-tuning the model with additional training on new data may be a more effective approach.

![alt text](image-2.png)

----
**<u>Transformers</u>:**

The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively.

The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically. 

The Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting. In addition, the authors introduce a positional encoding scheme that encodes the position of each token in the input sequence, enabling the model to capture the order of the sequence without the need for recurrent or convolutional operations.

![alt text](image.png)
