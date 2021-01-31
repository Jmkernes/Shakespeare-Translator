# Pre-training

This entire folder can be ignored. After experimentation it was not found to increase performance, and ended up making the model much larger and slower than it needed to be (well not that much larger, about a 1/3 increase from ~22M->~31M parameters)

We outline the general flow of how this works however, for completeness. In this directory we perform a masked language modeling (LM) task on text corresponding to the inputs only. We employ a cloze task, and use the BERT pretraining parameters, but do not do the next sentence prediction task that they do. Thus, we don't need to keep track of [sep] or [cls] tags.

We randomly select 15% of tokens to mask, and 80% of the time replace them with a [MASK] token, and 10% of the time with a random token (do nothing the other 10%). We then train the model to predict the missing tokens given the masked input. The loss is only computed over masked tokens.

The cloze data is generated on the fly, so that every epoch contains new data. Two models were trained, and are saved in dmod... zip files (the names give the parameters for d_model, num_heads, d_ffn, num_layers). Training logs are in the logs.zip file. We reached an accuracy of around 36% after 30 epochs with the larger model.

Another difference, is that we tie the weights during training. The same weights used for embedding are used for the final dense layer as well.
