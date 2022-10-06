# Text Generation Project: Instructions

This file collects the original Udacity instructions for the submission of the project.

## Requirements

- Include all project files; see [submission](#submission).
- Pass all unit tests.
- Function `create_lookup_tables()` creates `vocab_to_int` and `int_to_vocab`.
- Tokenization function: `token_lookup()`.
- Data loaders created which deliver sequences of appropriate length and the desired batches.
- RNN model fully defined:
  - All functions: `init`, `forward`, `init_hidden`.
  - At least one LSTM/GRU and at least one fully connected layer.
- Training with reasonable hyperparameters:
  - Enough epochs, until training loss plateaus.
  - Batch size large enough to train efficiently given the memory constraints.
  - Embedding dimension: much smaller than vocabulary size.
  - Hidden dimension: large enough.
  - Number of GRU/LSTM layers: 1-3.
  - `seq_length`: around the length of the sentences looked to guess next word.
  - Learning rate: small enough (to learn properly), but not too small (slow).
- Learning curves should decrease, we should get a loss below 3.5
- Justify hyperparameter choice.
- Generated text should look structurally similar to training text, but it doesn't need to make sense.

Suggestions:

- Use validation data to choose the best model.
- Initialize your model weights, especially the weights of the embedded layer to encourage model convergence.
- Use top-k sampling to generate new words.

## Training

Keep GPU workspace running longer than 30 minutes with the utility function `active_session()` called with the context manager:

```python
from workspace_utils import active_session

with active_session():
    # do long-running work here
```

If we get the `Out of memory` error:

- decrease the `batch_size`
- restart the kernel to remove old unused objects from memory

## Submission

- Pass all unit tests.
- Save notebook as HTML.
- ZIP the files:
  - HTML notebook
  - IPYNB notebook
  - `helper.py`
  - `problem_unittests.py`
