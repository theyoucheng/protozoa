python sbfl.py --model ../saved_models/cifar10_complicated.h5 --inputs ../../../../Dropbox/github/sbte/src/data-cifar10/ --cifar10-dataset --testgen-factor 0.2 --testgen-size 2000 --measure zoltar --outputs outs2

python sbfl.py --mobilenet-model --inputs ../data/  --testgen-size 200 --measure zoltar --outputs outs-mobilenet --testgen-factor 0.2
