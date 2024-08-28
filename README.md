# T5


T5 leverages the Transformer architecture and treats tasks like translation, summarization, classification, and others under a unified framework where the task is defined as a sequence of input text converted into an output text.


T5 is based on the Transformer architecture, which consists of an encoder-decoder structure:
Encoder: Processes the input text sequence.
Decoder: Generates the output text sequence.

The input to T5 is a sequence of tokens, represented as:
X = [x_1, x_2, ..., x_n]
where x_i is the token ID of the i-th token in the input sequence.

Since the Transformer architecture does not inherently capture the order of tokens, T5 adds positional encoding to the token embeddings to include information about the position of tokens in the sequence.

The positional encoding for the i-th token at dimension k is given by:
PE(i, 2k) = sin(i / 10000^(2k/d_model))
PE(i, 2k+1) = cos(i / 10000^(2k/d_model))
Where:
i is the position of the token.
k is the dimension of the embedding.
d_model is the dimensionality of the model.

The self-attention mechanism allows the model to focus on different parts of the input sequence simultaneously, which is a key component of the Transformer.

Given an input sequence X, three matrices are derived:
1) Query (Q):
Q = X * W_Q
2) Key (K):
K = X * W_K
3) Value (V):
V = X * W_V
Where:
W_Q, W_K, and W_V are learned weight matrices.

The attention scores are calculated using:
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
Where:
d_k is the dimension of the key vectors.
softmax normalizes the attention scores.

In T5, multi-head attention allows the model to project the input sequence into multiple subspaces, perform attention independently in each subspace, and then combine the results.
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
Where:
head_i = Attention(Q_i, K_i, V_i) for each head i.
W_O is a learned weight matrix that projects the concatenated outputs.

Each token's representation is then passed through a feed-forward neural network, applied identically and independently to each position:
FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
Where:
W_1, W_2 are learned weight matrices.
b_1, b_2 are learned biases.
max(0, x) is the ReLU activation function.

Layer normalization is applied to the outputs of both the multi-head attention and the feed-forward network. Residual connections are used to add the original input back to the output of these layers:
Z = LayerNorm(X + MultiHead(Q, K, V))
Output = LayerNorm(Z + FFN(Z))

The encoder and decoder in T5 are stacks of these Transformer blocks:
Encoder: Encodes the input sequence into a context-aware representation.
Decoder: Decodes the encoded representation into the output sequence, using self-attention and cross-attention with the encoder's output.

T5's innovation is in framing all tasks as text-to-text:
Input: A task-specific prompt followed by the input text.
"translate English to French: How are you?"
Output: The corresponding text output.
"Comment ça va?"
During training, T5 is provided with paired input-output sequences and learns to map the input text to the correct output text.

The loss function used in T5 is typically the cross-entropy loss, which measures the difference between the predicted and true token sequences:
Loss = -sum(y_true * log(y_pred))
Where:
y_true is the true token sequence.
y_pred is the predicted token sequence (output of the model after softmax).

Once pre-trained on a large corpus, T5 can be fine-tuned on specific tasks by adjusting the task-specific input-output pairs. The text-to-text framework simplifies this transfer learning process.
