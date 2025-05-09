import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import re

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

# %% -------------------------------------------------------------------------

# ##########################################################################
# # Configuration
# ##########################################################################

batch_size = 64
context_length = 12
n_embeddings = 20
n_layers = 2
n_heads = 2
dropout = 0.2
eval_iterations = 100

# %% -------------------------------------------------------------------------

# ###########################################################################
# # Data Loading and Tokenization
# ###########################################################################

# load data
with open("./data-train.txt", "r") as f:
    input_text = f.read()

# configure tokenizer
token_regex = r"\w+|\._|,"
input_tokens = re.findall(token_regex, input_text)
vocab = set(list(input_tokens))

stoi = {word: i for i, word in enumerate(vocab)}

unknown_token_string = "UNKNOWN"
stoi[unknown_token_string] = len(stoi)
pad_token_string = "PADDING"
stoi[pad_token_string] = len(stoi)

vocab_size = len(stoi)
print(f"Vocab: {vocab_size} tokens. {list(vocab)[:10]}")

itos = {i: word for word, i in stoi.items()}


def encode(string):
    s = string.lower()
    token_texts = re.findall(token_regex, s)
    unknown_index = stoi[unknown_token_string]
    return [stoi.get(token_text, unknown_index) for token_text in token_texts]


decode = lambda index_list: " ".join([itos[index] for index in index_list])


# tokenize dataset
input_text_lines = re.findall(r".*\n", input_text)
input_token_lines = list(map(lambda l: encode(l), input_text_lines))
max_line_length = max([len(line) for line in input_token_lines]) + 1

if max_line_length > context_length:
    for i in range(10):
        print("\n\n\n************** context length too small *************\n\n\n\n")

input_token_lines_padded = torch.stack(
    [
        F.pad(
            torch.tensor(line, device=device),
            (0, context_length - len(line)),
            value=stoi[pad_token_string],
        )
        for line in input_token_lines
    ]
)

# load and tokenize validation set
with open("./data-valid.txt") as f:
    val_text = f.read()

val_tokens = encode(val_text)


# ############################################################################
# # Utility Functions
# ############################################################################


@torch.no_grad()
def validate():
    input = torch.tensor([val_tokens[:-1]], device=device)
    targets = torch.tensor([val_tokens[1:]], device=device)
    _, loss = model.forward(input, targets)
    if loss is None:
        raise ValueError("Loss is None")
    return loss.item()


@torch.no_grad()
def get_batch():
    rnd = torch.randint(0, len(input_token_lines_padded), (batch_size,), device=device)
    x = torch.stack([t[:-1] for t in input_token_lines_padded[rnd]])
    y = torch.stack([t[1:] for t in input_token_lines_padded[rnd]])
    return x, y


@torch.no_grad()
def get_loss(model):
    loss = 0
    model.train(False)
    for _ in range(eval_iterations):
        x, y = get_batch()
        _, l = model(x, y)
        loss += l
    model.train()

    return loss / eval_iterations


# %% -------------------------------------------------------------------------

# ##########################################################################
# # Model
# ##########################################################################


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        # transforms an incoming embedding into a key (information to share) head_size dimensional vector
        self.key = nn.Linear(n_embeddings, head_size, bias=False)

        # transforms an incoming embedding into a query (information to look for) head_size dimensional vector
        self.query = nn.Linear(n_embeddings, head_size, bias=False)

        # transforms an incoming embedding into a head_size dimensional vector called value
        self.value = nn.Linear(n_embeddings, head_size, bias=False)

        # buffer to mask attention scores for future values
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, seq_len, _ = x.shape  # (batch, seq_len, n_embeddings)

        key = self.key(x)  # (batch, seq_len, head_size)
        query = self.query(x)  # (batch, seq_len, head_size)
        value = self.value(x)  # (batch, seq_len, head_size)

        # compute attention scores ("affinities")
        # (batch, seq_len, head_size) @ (batch, head_size, seq_len) -> (batch, seq_len, seq_len)
        weights = query @ key.transpose(-2, -1)

        # scale
        weights /= math.sqrt(key.shape[-1])

        # mask attention scores for future values
        tril = self.get_buffer("tril")
        weights = weights.masked_fill(
            tril[:seq_len, :seq_len] == 0, float("-inf")
        )  # (batch, seq_len, seq_len)

        weights = F.softmax(weights, dim=-1)  # (batch, seq_len, seq_len)

        # Store weights before dropout for visualization
        self.last_weights = weights  # (batch, seq_len, seq_len)

        weights_after_dropout = self.dropout(weights)

        # perform weighted aggregation of values
        # (batch, seq_len, seq_len) @ (batch, seq_len, head_size) -> (batch, seq_len, head_size)
        out = weights_after_dropout @ value

        return out


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(
            n_heads * head_size, n_embeddings
        )  # TODO: n_embeddings == n_heads * head_size?
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]  # Run heads

        # Store attention weights from each head for visualization
        # Check hasattr in case a head implementation changes or during model loading issues
        self.last_attention_weights = [
            head.last_weights.detach().cpu().numpy()
            for head in self.heads
            if hasattr(head, "last_weights")
        ]

        out = torch.cat(head_outputs, dim=-1)  # Concatenate results
        out = self.projection(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_embeddings, n_heads):
        super().__init__()

        head_size = n_embeddings // n_heads

        self.layer_norm1 = nn.LayerNorm(n_embeddings)
        self.self_attn = MultiheadAttention(n_heads, head_size)

        self.layer_norm2 = nn.LayerNorm(n_embeddings)
        self.feed_forward = nn.Sequential(
            nn.Linear(
                n_embeddings, 4 * n_embeddings
            ),  # TODO: what is this hardcoded 4?
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # first layer norm, then self-attention, plus residual connection
        x = x + self.self_attn(self.layer_norm1(x))

        # first layer norm, then feed forward, plus residual connection
        x = x + self.feed_forward(self.layer_norm2(x))

        # Store the main output of the TransformerBlock module for visualization
        self.last_output = x.detach().cpu().numpy()

        return x


class GPTLanguageModelSnapshot:
    embeddings = []
    transformer_blocks = []
    attention_data = []


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.snapshots = GPTLanguageModelSnapshot()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(context_length, n_embeddings)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embeddings,
                    n_heads,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(n_embeddings)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, indices, targets=None):
        batch_size, seq_len = indices.shape

        token_embeddings = self.token_embedding_table(indices)

        positions = torch.arange(seq_len, device=device)
        position_embeddings = self.position_embedding_table(positions)

        x = token_embeddings + position_embeddings  # (batch, tokens, n_embeddings)
        x = self.transformer_blocks(x)  # (batch, tokens, n_embeddings)
        x = self.final_layer_norm(x)  # (batch, tokens, n_embeddings)

        # TODO: in inference mode, only calc logits for last token
        logits = x @ self.token_embedding_table.weight.T

        # training
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_len, vocab_size)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)

        # inference
        else:
            loss = None

        # get snapshots of the first batch for visualization during inference
        if targets is None:
            self.snapshots.embeddings = token_embeddings[0].detach().cpu().numpy()
            self.snapshots.transformer_blocks = []
            self.snapshots.attention_data = []
            for block in self.transformer_blocks:
                self.snapshots.transformer_blocks.append(block.last_output[0])
                self.snapshots.attention_data.append(
                    [w[0] for w in block.self_attn.last_attention_weights]
                )

        return logits, loss

    def generate(self, indices, max_new_tokens, top_k=3, token_probs=[]):

        # number of new tokens times do
        for _ in range(max_new_tokens):

            # crop the current context to max context size
            cropped_context = indices[:, -context_length:]

            # forward the model on the context
            logits, _ = self(cropped_context)

            # get only the last token logits
            logits = logits[:, -1, :]

            # transform logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (batch, vocab_size)

            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

            # debug
            token_probs.clear()
            token_probs += [
                *(
                    zip(
                        # squeeze out batch dimension
                        top_indices.squeeze(dim=0).tolist(),
                        top_probs.squeeze(dim=0).tolist(),
                    )
                )
            ]

            # sample next token from the top k
            next_of_topk = torch.multinomial(top_probs, num_samples=1)  # (batch, 1)

            # get the index of the sampled token
            next = top_indices[:, next_of_topk[0]]

            # append sampled token to the sequence
            indices = torch.cat((indices, next), dim=1)  # batch_size, seq_len+1

        return indices


# #############################################################################
# # Generate utility function with debug
# #############################################################################


def generate(str, max_new_tokens=1):
    print("Input:", decode(encode(str)))
    context_indices = torch.tensor([encode(str)], device=device)

    result = ""
    for _ in range(max_new_tokens):
        debug: list[tuple[int, float]] = []
        context_indices = model.generate(
            context_indices, max_new_tokens=1, top_k=5, token_probs=debug
        )

        word = itos[(context_indices[0].tolist()[-1])]
        print(f"Output: {word}")
        result += f" {word}"

        for d in sorted(debug, key=lambda x: x[1], reverse=True):
            print(f"{itos[d[0]]:20s} {d[1]*100:.2f}%")

    return result


# ##############################################################################
# # Model instanciation
# ##############################################################################

model = GPTLanguageModel()
model.to(device)
# print(f"{model=}")
parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {(parameter_count)/(1024*1024):.3f}M ({parameter_count})")


# %% -------------------------------------------------------------------------

# ##############################################################################
# # Train the model
# ##############################################################################

learning_rate = 0.0001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training_iterations = 1 # debug
training_iterations = 10000

losses = []

_ = model.train()
for i in tqdm(range(training_iterations)):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    losses.append(loss.item())

    if i % 500 == 0:
        tqdm.write(f"train loss: {losses[-1]:.3f}")
        tqdm.write(f"valid loss: {validate():.3f}")
        _ = model.eval()
        generate("i like spicy so i like")
        _ = model.train()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# set the model to eval mode, e.g. disable dropout
_ = model.train(False)

tqdm.write(f"train loss: {losses[-1]:.3f}")
tqdm.write(f"valid loss: {validate():.3f}")

_ = generate("i like sour so i like")
_ = generate("i like juicy so i like")
_ = generate("i like spicy so i like")

# %% -------------------------------------------------------------------------

from visualization import view_transformer_and_attention

# %% -------------------------------------------------------------------------

# ###########################################################################
# # Embedding visualization
# ###########################################################################

texts = [
    # "juicy sour sweet spicy",
    "lemon apple orange chili",
    # "chili"
    # "chili lemon taste apple"
]

for text in texts:
    input_token_texts = [itos[i] for i in encode(text)]
    tokens = encode(text)
    embeddings = (
        model.token_embedding_table(torch.tensor(tokens, device=device))
        .detach()
        .cpu()
        .numpy()
    )

    # Embeddings only snapshot
    vis = GPTLanguageModelSnapshot()
    vis.embeddings = embeddings

    view_transformer_and_attention(
        snapshot=vis,
        input_token_texts=input_token_texts,
    )

# %% -------------------------------------------------------------------------

# ###########################################################################
# # Forward pass visualization
# ###########################################################################

text = "i like spicy so i like"
print(decode(encode(text)))
generate(text, max_new_tokens=1)

context_indices = torch.tensor([encode(text)], device=device)
_ = model(context_indices)  # Run forward pass to pupulate snapshots

input_token_texts = [itos[i] for i in encode(text)]
view_transformer_and_attention(
    snapshot=model.snapshots,
    input_token_texts=input_token_texts,
)
