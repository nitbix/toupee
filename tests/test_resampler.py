import numpy
import toupee
from toupee.data import WeightedResampler

class TestResampler:

    def __init__ (self):
        x = numpy.asarray(range(4))
        y = numpy.asarray(range(4))
        self.dataset = ((x,y),(x,y),(x,y))
        self.weights = [0.1,0.4,0.4,0.1]

    def test_weighted_resampler(self):
        r = WeightedResampler(self.dataset)
        print numpy.asarray(r.make_new_train(10))
        r.update_weights(self.weights)
        print numpy.asarray(r.make_new_train(10))

if __name__ == "__main__":
    r = TestResampler()
    r.test_weighted_resampler()
