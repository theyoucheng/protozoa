# Protozoa

Protozoa is a tool to explain output from the neural network image classifier.

##To start

```
python3 ./src/protozoa.py --mobilenet-model --inputs data/ --outputs outs
```

To try another pre-trained neural network model: ``--xception-model'', ``--vgg16-model''.

Alternatively, you are able to use your own model, by giving it to ``--model MODEL'',
in which case you also need to tell Protozoa the input image format: the number of rows ``--input-rows INT'' and columns ``--input-cols INT''.

By default, Protozoa generates 2,000 mutants per explanation, you can configure this number by e.g., ``--testgen-size 200'' for more efficient run.
