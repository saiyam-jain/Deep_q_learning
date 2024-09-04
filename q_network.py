import tensorflow as tf

# Define the Q-Network
class QNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(128)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(128)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(output_dim, activation='linear')

    def call(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.dropout3(x)
        return self.out(x)