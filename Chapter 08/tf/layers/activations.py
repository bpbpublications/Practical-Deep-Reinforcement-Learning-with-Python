import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-10.0, 10.0, 1000)

# ReLU
plt.title('ReLU')
plt.plot(x.numpy(), tf.nn.relu(x).numpy())
plt.show()

# Sigmoid
plt.title('Sigmoid')
plt.plot(x.numpy(), tf.nn.sigmoid(x).numpy())
plt.show()

# Tanh
plt.title('Tanh')
plt.plot(x.numpy(), tf.nn.tanh(x).numpy())
plt.show()
