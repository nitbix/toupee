#This scripts assumes we are running from toupee's test folder
TIN200_FOLDER=/datasets/tinyimagenet_200


echo TODO: create S3 bucket for this dataset

#if the data folder doesn't exist, creates it
if [ ! -d "$TIN200_FOLDER" ]; then
    
    echo Dataset not found, copying from S3...
    
    if [ ! -d "/datasets" ]; then 
        sudo mkdir /datasets
        sudo chmod 777 /datasets
    fi
    
    mkdir $TIN200_FOLDER
    chmod 777 $TIN200_FOLDER
    aws s3 cp s3://public-datasets.nplan.io/tinyimagenet_200/test.npz $TIN200_FOLDER
    aws s3 cp s3://public-datasets.nplan.io/tinyimagenet_200/train.npz $TIN200_FOLDER
    aws s3 cp s3://public-datasets.nplan.io/tinyimagenet_200/valid.npz $TIN200_FOLDER
    aws s3 cp s3://public-datasets.nplan.io/tinyimagenet_200/AdaBoost.yaml $TIN200_FOLDER
fi

KERAS_BACKEND=theano THEANO_FLAGS=device=cuda7 python3 ../bin/ensemble.py $TIN200_FOLDER/AdaBoost.yaml --test-tin200 --no-dump
