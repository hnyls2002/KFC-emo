#### exp-2

- batch_size = 64
- num_epochs = 10
- learning_rate = 5e-5
- dropout_prob = 0.1
- threshold = 0.5

```py
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer=optimizer, base_lr=1e-5, max_lr=5e-5, step_size_up=batch_nums / 3, step_size_down=batch_nums * 2 / 3, mode="triangular2" , gamma=0.9, cycle_momentum=False)
```

```
Epoch 5, Dev Loss: 0.11958104, Dev Acc: 20.3991%, F1 Score: 22.0858%
f1s :  [0.45514563 0.52361331 0.2115573  0.00951684 0.13137973 0.12658228
 0.08641202 0.20358306 0.2605364  0.0189911  0.07024467 0.1588386
 0.17761989 0.11697807 0.34716157 0.80958781 0.         0.27425373
 0.64185229 0.         0.34440344 0.         0.02885683 0.
 0.22814815 0.29809725 0.27651007 0.38416527]
```