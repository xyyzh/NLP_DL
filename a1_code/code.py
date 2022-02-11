from typing import Union, Iterable, Callable
import random

import torch.nn as nn
import torch
import math

def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    return {
        word: ix
        for ix, (word, _) in enumerate(
            sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[
                :max_words
            ]
        )
    }


def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


### 1.1 Batching, shuffling, iteration
def build_loader(
    data_dict: dict, batch_size: int = 64, shuffle: bool = False
) -> Callable[[], Iterable[dict]]:

    """
    Build a nested function. build_loader(...) specifies what type of loader
    you want, and the output is itself a function that, when called, returns
    a generator. You can iterate over the generator to get a batch,
    which is a dictionary with the same keys, and values are lists of
    length batch_size (the last batch may be shorter since you only need
    to include the remaining samples).

    When shuffle=True, then every time you iterate through loader, the batches 
    will contain different samples. This means the order of the training set is 
    randomized every time you call `for batch in loader()`

    Parameters
    ----------
    data_dict: dict
        A dictionary with keys 'premise', 'hypothesis', and potentially
        'label', all of which are lists of same length.
    batch_size: int, optional
        The size of the batch. The length of the list in the batch yield by
        loader will be equal to batch_size, except for the last batch, which
        may be shorter (since it contains the remaining samples).
    shuffle: bool, optional
        Whether to shuffle the dataset.

    Returns
    -------
    function
        A loader generator with no input and returns an iterator yielding a
        dictionary with the same keys as data_dict, but with values of length
        corresponding to batch_size (or shorter if the last batch is shorter).

    Notes
    -----
    It's possible to implement this function such that data_dict could have
    arbitrary keys as long as they are all lists of same length.

    Examples
    --------
    >>> loader = build_loader(data)  # let's assume 300 samples
    >>> for batch in loader():
    ...     premise = batch['premise']
    ...     label_batch = batch['label']
    ...     # do something with batch here
    ...     print(len(premise))
    64
    64
    64
    8
    """
    # TODO: Your code here
        
    def loader():
        # TODO: Your code here
        if shuffle:
            premise = data_dict['premise']
            hypothesis = data_dict['hypothesis']
            label = data_dict['label']
            # Make sure they are in correspondance
            instance = list(zip(premise, hypothesis, label))
            random.shuffle(instance)
            premise, hypothesis, label = zip(*instance)
            data_dict['premise'] = premise
            data_dict['hypothesis'] = hypothesis
            data_dict['label'] = label
            
        # batches = list()
        # example: 200 samples, batch_size=64
        # num_batch = 3  i=0,1,2
        # [0:64] [64:128] [128:192]
        # [192:]
        
        # example 2: 192 samples, batch_size=64
        # num_batch = 3  i=0,1,2
        # [0:64] [64:128] [128:192]
        num_batch = len(data_dict['premise'])//batch_size
        for i in range(num_batch):
            one_batch = {'premise': data_dict['premise'][batch_size*i:batch_size*(i+1)],
                        'hypothesis': data_dict['hypothesis'][batch_size*i:batch_size*(i+1)],
                        'label': data_dict['label'][batch_size*i:batch_size*(i+1)]}
            yield one_batch
        if batch_size*num_batch < len(data_dict['premise']):
            remaining = {'premise': data_dict['premise'][batch_size*num_batch:],
                            'hypothesis': data_dict['hypothesis'][batch_size*num_batch:],
                            'label': data_dict['label'][batch_size*num_batch:]}
            yield remaining

    return loader

### 1.2 Converting a batch into inputs
def convert_to_tensors(text_indices: "list[list[int]]") -> torch.Tensor:
    """
    Given a list of lists of indices, convert it to a tensor of shape (N, L),
    You will need to handle the padding, which will be of value 0.

    Parameters
    ----------
    text_indices: list of list of int
        A list of token indices, which can be either the premise or hypothesis
        from a batch yield by loader().
    
    Returns
    -------
    torch.Tensor of torch.int32
        A tensor of shape (N, L) where L is the length of the longest inner list, 
        and N is the length of the outer list.
    """
    # TODO: Your code here
    text_length = [len(lst) for lst in text_indices]
    max_length = max(text_length)
    for i in range(len(text_indices)):
        text_indices[i] += [0] * (max_length - len(text_indices[i]))
        
    text_indices = torch.tensor(text_indices, dtype=torch.int32)
    return text_indices


### 2.1 Design a logistic model with embedding and pooling
def max_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Take the pooling over the second dimension, i.e. a
    (N, L, D) -> (N, D) transformation where D is the `hidden_size`,
    N is the batch size, L is the sequence length.
    """
    # TODO: Your code here
    return torch.max(x,1).values

class PooledLogisticRegression(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        """
        When called this simple linear model will do the following:
            1. Individually embed a batch of premise and hypothesis (token indices)
            2. Individually apply max_pool along the sequence length (L_p and L_h)
            3. Concatenate the pooled tensors into a single tensor
            4. Apply the logistic regression to obtain prediction

        Parameters
        ----------
        embedding: nn.Embedding
            The embedding layer you created using the size of the word index.
            You can create it outside of this module. The transformation is
            (N, L) -> (N, L, E) where E is the initial embedding dimension, and L is
            the sequence length.
        """
        super().__init__()
        self.embedding = embedding
        E = self.embedding.weight.shape[1]
        self.layer_pred = nn.Linear(2*E, 1)
        self.sigmoid = nn.Sigmoid()
        # TODO: Your code here

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        premise: torch.Tensor[N, L_p]
            The premise tensor, where L_p is the premise sequence length and
            N is the batch size.
        hypothesis: torch.Tensor[N, L_h]
            The hypothesis tensor, where L_h is the hypothesis sequence length.

        Returns
        -------
        torch.Tensor[N]
            The predicted score for each example in the batch.

        Notes
        -----
        Note the returned tensor is of shape N, not (N, 1). You will need to
        reshape your tensor to get the correct format.
        """
        
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()

        # TODO: Your code here
        prem = max_pool(emb(premise))
        hypo = max_pool(emb(hypothesis))
        concat = torch.cat((prem, hypo), 1)
        concat = layer_pred(concat)
        output = sigmoid(concat)
        output = torch.squeeze(output)
        return output


### 2.2 Choose an optimizer and a loss function
def assign_optimizer(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    """
    Parameters
    ----------
    model: nn.Module
        The model to optimize.
    kwargs: dict
        The arguments to pass to the optimizer. This will vary depending on the
        optimizer, but the most common one is `lr`.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer that you will use during the model training.

    Notes
    -----
    There's many optimizers in PyTorch. You can start with SGD, but
    it's recommended to try other popular options:
    https://pytorch.org/docs/stable/optim.html#algorithms
    """
    # TODO: Your code here
    return torch.optim.Adam(model.parameters(), **kwargs)



def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    The binary cross entropy loss, implemented from scratch using torch
    Do not use torch.nn, but you may compare your implementation against
    the official one.

    Parameters
    ----------
    y: torch.Tensor[N]
        The true labels.
    y_pred: torch.Tensor[N]
        The predicted labels.

    Returns
    -------
    torch.Tensor
        The binary cross entropy loss (averaged over N).
    """
    # TODO: Your code here
    entropy_sum = 0
    for true, pred in zip(y, y_pred):
        term1 = true * math.log(pred + 1e-7)
        term2 = (1 - true) * math.log(1 - pred + 1e-7)
        entropy_sum += (term1 + term2)
    return -entropy_sum / len(y)


### 2.3 Forward and backward pass
def forward_pass(model: nn.Module, batch: dict, device="cpu"):
    """
    Implement a function that performs one step of the training process. Given
    a batch and a model, this function should handle the text to tensor conversion
    and pass it in a model.

    Parameters
    ----------
    model: nn.Module
        The model you will use to perform the forward pass.
    batch: dict of list
        A dictionary with 'premise' and 'hypothesis' keys (lists of same size).
    device: str
        The device you want to run the model on. This is usually 'cpu' or 'cuda'.

    Returns
    -------
    torch.Tensor
        The predicted labels.

    This function should return the predicted y value by the model.
    """
    # TODO: Your code here
# =============================================================================
#     prem_tokens = tokenize(batch['premise'])
#     hypo_tokens = tokenize(batch['hypothesis'])
#     prem_ix = tokens_to_ix(prem_tokens, build_index_map(build_word_counts(prem_tokens)))
#     hypo_ix = tokens_to_ix(hypo_tokens, build_index_map(build_word_counts(hypo_tokens)))
# =============================================================================
    prem_tensor = convert_to_tensors(batch['premise']).to(device)
    hypo_tensor = convert_to_tensors(batch['hypothesis']).to(device)
    model.to(device)
    return model.forward(prem_tensor, hypo_tensor)



def backward_pass(
    optimizer: torch.optim.Optimizer, y: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    This function takes in the optimizer, the true labels, and the predicted labels,
    then computes the loss and performs a backward pass before updating the weights.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer you will use to perform the backward pass.
    y: torch.Tensor[N]
        The true labels.
    y_pred: torch.Tensor[N]
        The predicted labels.

    Returns
    -------
    torch.Tensor
        The loss value computed with bce_loss()
    """
    # TODO: Your code here
    loss = bce_loss(y, y_pred)
    #loss.backward()  #there's no backward method for the loss？
    optimizer.step()
    optimizer.zero_grad()
    return loss


### 2.4 Evaluation
def f1_score(y: torch.Tensor, y_pred: torch.Tensor, threshold=0.5) -> torch.Tensor:
    """
    Apply the threshold (if it is not None), then compute the F1 score from scratch 
    (without using external libraries).

    Parameters
    ----------
    y: torch.Tensor[N]
        The true labels.
    y_pred: torch.Tensor[N]
        The predicted labels.
    threshold: float, default 0.5
        The threshold to use to convert the predicted labels to binary. If set
        to None, y_pred will not be thresholded (in this case, we assume y_pred
        is already binary).

    Returns
    -------
    torch.Tensor[1]
        The F1 score.

    """
    # TODO: Your code here
    if (threshold):
        y_pred = y_pred.apply_(lambda x: 1 if x >= threshold else 0)
    tp = torch.sum((y == y_pred) * (y_pred == 1))
    fp = torch.sum((y != y_pred) * (y_pred == 0))
    fn = torch.sum((y != y_pred) * (y_pred == 1))
    return tp / (tp + (fp + fn) / 2)


### 2.5 Train loop
def eval_run(
    model: nn.Module, loader: Callable[[], Iterable[dict]], device: str = "cpu"
):
    """
    Iterate through a loader and predict the labels for each example, all while
    collecting the original labels.

    Parameters
    ----------
    model: nn.Module
        The model you will use to perform the forward pass.
    loader: Callable[[], Iterable[dict]]
        The loader function that will yield batches.
    device: str
        The device you want to run the model on. This is usually 'cpu' or 'cuda'.

    Returns
    -------
    Return the true labels and predicted ones for all data samples at once.
    The length should correspond to the length of the input labels (before you give to the loader).
    y_true: torch.Tensor[N]
        The true labels, extracted from the loader.
    y_pred: torch.Tensor[N]
        The labels predicted by the model (output of forward_pass).

    Notes
    -----
    You can use the `forward_pass` function to get the predicted labels. Don't
    forget to  disable the gradients for the model and to turn your model into
    evaluation mode.
    """
    # TODO: Your code here
    y_true = list()
    y_pred = list()
    model.eval()
    for batch in loader():
        with torch.no_grad(): 
            label = batch['label']
            y_true.extend(label)
            pred_label = forward_pass(model, batch, device)
            y_pred.extend(pred_label.tolist())
    y_true = torch.tensor(y_true, dtype=torch.float)
    y_pred = torch.tensor(y_pred, dtype=torch.float)
    return y_true, y_pred


def train_loop(
    model: nn.Module,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs: int = 3,
    device: str = "cpu",
):
    """
    Train a model for a given number of epochs.

    Parameters
    ----------
    model: nn.Module
        The model you will use to perform the forward pass.
    train_loader: Callable[[], Iterable[dict]]
        The loader function that will yield shuffled batches of training data.
    valid_loader: Callable[[], Iterable[dict]]
        The loader function that will yield non-shuffled batches of validation data.
    optimizer: torch.optim.Optimizer
        The optimizer you will use to perform the backward pass.
    n_epochs: int
        The number of epochs you want to train your model
    device: str
        The device you want to run the model on. This is usually 'cpu' or 'cuda'.

    Returns
    -------
    list
        A list of f1 scores evaluated on the valid_loader at the end of each epoch.

    Notes
    -----
    This function is left open-ended and is strictly to help you train your model.
    You are free to implement what you think works best, as long as it runs on the
    training and validation data and return a list of validation score at the end
    of each epoch.
    """
    # TODO: Your code here
    
    f1 = list()
    for epoch in range(n_epochs):
        print('================ Epoch {} / {} ================'.format(epoch + 1, n_epochs))  
        print('Training...')
        model.train()     
        for batch in train_loader():
            #model.zero_grad()
            label = batch['label']
            y_true = torch.tensor(label, dtype=torch.float)
            y_pred = forward_pass(model, batch, device)    
            #print("y_true and y_pred ", y_true, y_pred)
            train_loss = backward_pass(optimizer, y_true, y_pred)
        
        #y_true, y_pred = eval_run(model, train_loader, device)
        
        print("Training loss is ", train_loss.item())
        
        print("Running Validation...")
        model.eval()
        train_y_true, train_y_pred = eval_run(model, train_loader, device)
# =============================================================================
        #print(train_y_true, train_y_pred)
# =============================================================================
        train_score = f1_score(train_y_true, train_y_pred)
        print("F1-score for training is ", train_score.item())
        
        val_y_true, val_y_pred = eval_run(model, valid_loader, device)
        val_score = f1_score(val_y_true, val_y_pred)
        print("F1-score for validation is ", val_score.item())
        f1.append(val_score)
    return f1
        



### 3.1
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int):
        """
        When called this simple linear model will do the following:
            1. Individually embed a batch of premise and hypothesis (token indices)
            2. Individually apply max_pool along the sequence length (L_p and L_h)
            3. Individually apply one feedforward layer to your pooled tensors
            4. Use the ReLU on the outputs of your layer
            5. Concatenate the activated tensors into a single tensor
            6. Apply sigmoid layer to obtain prediction

        Parameters
        ----------
        embedding: nn.Embedding
            The embedding layer you created using the size of the word index.
        hidden_size: int
            The size of the hidden layer.
        """
        super().__init__()

        # TODO: continue here
        self.embedding = embedding
        E = self.embedding.weight.shape[1]
        self.ff_layer = nn.Linear(2*E, hidden_size) 
        self.activation = nn.ReLU()
        self.layer_pred = nn.Linear(hidden_size, 1) 
        #self.layer_pred = nn.Linear(2*E, 1)
        self.sigmoid = nn.Sigmoid()
        

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layer(self):
        return self.ff_layer

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        premise: torch.Tensor[N, L_p]
            The premise tensor, where N is the batch size and L_p is the premise
            sequence length.
        hypothesis: torch.Tensor[N, L_h]
            The hypothesis tensor, where L_h is the hypothesis sequence length.

        Returns
        -------
        torch.Tensor[N]
            The scores for each example in the batch.
        """

        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layer = self.get_ff_layer()
        act = self.get_activation()

        # TODO: continue here
        prem = max_pool(emb(premise))
        hypo = max_pool(emb(hypothesis))
        concat = torch.cat((prem, hypo), 1)
        concat = ff_layer(concat)
        concat = act(concat)
        concat = layer_pred(concat)
        output = sigmoid(concat)
        output = torch.squeeze(output)
        return output


### 3.2
class DeepNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, num_layers: int = 2):
        """
        When called this simple linear model will do the following:
            1. Individually embed a batch of premise and hypothesis (token indices)
            2. Individually apply max_pool along the sequence length (L_p and L_h)
            3. Individually apply one feedforward layer to your pooled tensors
            4. Use the ReLU on the outputs of your layer, repeat (3) for `num_layers` times.
            5. Concatenate the activated tensors into a single tensor
            6. Apply sigmoid layer to obtain prediction

        Parameters
        ----------
        embedding: nn.Embedding
            The embedding layer you created using the size of the of the word index. You can
            create it outside of this module. The transforma dimensions is (N, L) -> (N, L, E) where
            E is the initial embedding dimension, and L is the sequence length.
        hidden_size: int
            The size of the hidden layer.
        num_layers: int, default 2
            The number of hidden layers in your deep network. Each layer must
            be activated with ReLU.

        Notes
        -----
        You will need to use nn.ModuleList to track your layers.
        """
        super().__init__()

        # TODO: continue here
        self.embedding = embedding
        E = self.embedding.weight.shape[1]
        self.activation = nn.ReLU()
        self.ff_layers = nn.ModuleList([nn.Linear(2*E, hidden_size)])
        # self.ff_layers = nn.Linear(2*E, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        for _ in range(num_layers-1):        
            self.ff_layers.append(middle_layer)
        
        #if num_layers > 1:
        #    for _ in range(num_layers):
        #        self.ff_layers = lambda x: self.middle_layer(self.activation(self.ff_layers(x)))
                
        self.layer_pred = nn.Linear(hidden_size, 1) 
        #self.layer_pred = nn.Linear(2*E, 1)
        self.sigmoid = nn.Sigmoid()

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layers(self):
        return self.ff_layers

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        premise: torch.Tensor[N, L_p]
            The premise tensor, where N is the batch size and L_p is the premise
            sequence length.
        hypothesis: torch.Tensor[N, L_h]
            The hypothesis tensor, where L_h is the hypothesis sequence length.

        Returns
        -------
        torch.Tensor[N]
            The scores for each example in the batch.
        """

        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layers = self.get_ff_layers()
        act = self.get_activation()

        # TODO: continue here
        prem = max_pool(emb(premise))
        hypo = max_pool(emb(hypothesis))
        concat = torch.cat((prem, hypo), 1)
        for l in ff_layers:
            concat = l(concat)
            concat = act(concat)
            
        #concat = ff_layers(concat)
        #
        concat = layer_pred(concat)
        output = sigmoid(concat)
        output = torch.squeeze(output)
        return output


if __name__ == "__main__":
    # If you have any code to test or train your model, do it BELOW!

    # Seeds to ensure reproducibility
    random.seed(2022)
    torch.manual_seed(2022)

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data")

    train_tokens = {
        "premise": tokenize(train_raw["premise"], max_length=64),
        "hypothesis": tokenize(train_raw["hypothesis"], max_length=64),
    }

    valid_tokens = {
        "premise": tokenize(valid_raw["premise"], max_length=64),
        "hypothesis": tokenize(valid_raw["hypothesis"], max_length=64),
    }

    word_counts = build_word_counts(
        train_tokens["premise"]
        + train_tokens["hypothesis"]
        + valid_tokens["premise"]
        + valid_tokens["hypothesis"]
    )
    index_map = build_index_map(word_counts, max_words=10000)

    train_indices = {
        "label": train_raw["label"],
        "premise": tokens_to_ix(train_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(train_tokens["hypothesis"], index_map)
    }

    valid_indices = {
        "label": valid_raw["label"],
        "premise": tokens_to_ix(valid_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(valid_tokens["hypothesis"], index_map)
    }

    # 1.1
    train_loader = build_loader(train_indices)
    valid_loader = build_loader(valid_indices)

    # 1.2
    batch = next(train_loader())
    y = batch['label']

    # 2.1
    embedding = nn.Embedding(500, 56)
    model = PooledLogisticRegression(embedding)

    # 2.2
    optimizer = assign_optimizer(model)

    # 2.3
    y_pred = "your code here"
    loss = "your code here"

    # 2.4
    score = "your code here"

    # 2.5
    n_epochs = 2

    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.1
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.2
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"
