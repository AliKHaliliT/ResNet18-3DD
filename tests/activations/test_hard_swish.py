import unittest
import tensorflow as tf
from ResNet183DD.assets.activations.hard_swish import HardSwish


class TestHardSwish(unittest.TestCase):

    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = HardSwish()(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


if __name__ == "__main__":
    unittest.main()