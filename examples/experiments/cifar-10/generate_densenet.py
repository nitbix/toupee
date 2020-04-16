import sys
import tensorflow as tf

LAYERS = 2

INPUT_SHAPE = (32,32,3)
OUTPUT_CLASSES = 10

def main():

    model = tf.keras.applications.DenseNet201(classes=OUTPUT_CLASSES,
                                              weights=None,
                                              input_shape=INPUT_SHAPE,
                                              #depth=190,
                                              #growth_rate=40
                                              )
    if len(sys.argv) >= 2:
        with open(sys.argv[1], 'w') as out:
            out.write(model.to_yaml())
        print("Model written to %s" % sys.argv[1])
    else:
        print(model.to_yaml())

if __name__ == "__main__":
    main()