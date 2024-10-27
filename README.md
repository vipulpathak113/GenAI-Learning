# Generative AI


**Generative AI:**

Generative AI is a type of AI that can create new content, such as text, images, music, audio, and videos. It's powered by large AI models, called foundation models, that can perform a variety of tasks, including summarization, Q&A, and classification.

---

**Prompt Engineering:**

Prompt engineering is a critical aspect of working with language models (LMs), such as GPT (Generative Pre-trained Transformer). It involves crafting the input text (the prompt) in a way that effectively guides the Language Models towards generating the desired output (the completion). This process can sometimes require multiple iterations of refinement to achieve the best results, a practice known as prompt engineering.

---

**Key Concepts in Language Model Interaction:**
- Prompt: The input text provided to the model.
- Inference: The process of generating text based on the prompt.
- Completion: The output text produced by the model.

-----


**Context Window:**
The context window refers to the total amount of text that the model can consider at one time. It is a critical factor since it limits the amount of information that can be used for generating responses.

----
**Transformers**:
The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively.

The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically. 

The Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting. In addition, the authors introduce a positional encoding scheme that encodes the position of each token in the input sequence, enabling the model to capture the order of the sequence without the need for recurrent or convolutional operations.

![alt text](image.png)
