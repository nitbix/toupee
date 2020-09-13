import sys
import tensorflow as tf
import densenet

LAYERS = 2

INPUT_SHAPE = (32,32,3)
OUTPUT_CLASSES = 10

DEPTH = 40
NB_DENSE = 3
GROWTH_RATE = 12
NB_FILTER = -1
DROPOUT = 0.0

def main():

    # model = tf.keras.applications.DenseNet121(classes=OUTPUT_CLASSES,
    #                                           weights=None,
    #                                           input_shape=INPUT_SHAPE,
    #                                           #depth=190,
    #                                           #growth_rate=40
    #                                           )
    model = densenet.DenseNet(input_shape=INPUT_SHAPE, classes=OUTPUT_CLASSES,
                              depth=DEPTH, nb_dense_block=NB_DENSE, growth_rate=GROWTH_RATE,
                              nb_filter=NB_FILTER, dropout_rate=DROPOUT, weights=None)
    if len(sys.argv) >= 2:
        with open(sys.argv[1], 'w') as out:
            out.write(model.to_yaml())
        print("Model written to %s" % sys.argv[1])
    else:
        print(model.to_yaml())

if __name__ == "__main__":
    main()