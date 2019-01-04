#This scripts assumes we are running from toupee's test folder
MNIST_FOLDER=/datasets/mnist_th

#if the data folder doesn't exist, creates it
if [ ! -d "$MNIST_FOLDER" ]; then
    
    echo Dataset not found, copying from S3...
    
    if [ ! -d "/datasets" ]; then 
        sudo mkdir /datasets
        sudo chmod 777 /datasets
    fi
    
    mkdir $MNIST_FOLDER
    chmod 777 $MNIST_FOLDER
    aws s3 cp s3://public-datasets.nplan.io/mnist/test.npz $MNIST_FOLDER
    aws s3 cp s3://public-datasets.nplan.io/mnist/train.npz $MNIST_FOLDER
    aws s3 cp s3://public-datasets.nplan.io/mnist/valid.npz $MNIST_FOLDER
    aws s3 cp s3://public-datasets.nplan.io/mnist/AdaBoost.yaml $MNIST_FOLDER
fi

KERAS_BACKEND=theano THEANO_FLAGS=device=cuda7 python3 ../bin/ensemble.py $MNIST_FOLDER/AdaBoost.yaml --test-mnist
