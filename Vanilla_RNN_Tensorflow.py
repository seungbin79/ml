import tensorflow as tf
import numpy as np

inputs = np.array([
    [[1, 2]]
])

tf.reset_default_graph()
tf.set_random_seed(777)
tf_inputs = tf.constant(inputs, dtype=tf.float32)
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=5)
outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, dtype=tf.float32, inputs=tf_inputs)
variables_names =[v.name for v in tf.trainable_variables()]

print(outputs)
print(state)
print("weights")
for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(v)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_run, state_run = sess.run([outputs, state])
    print("output values")
    print(output_run)
    print("\nstate value")
    print(state_run)
    print("weights")
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        print(k, v)



# I      [1,0,0,0]
# work   [0,1,0,0]
# at     [0,0,1,0]
# google [0,0,0,1]
#
# I work at google =  [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ]
# I google at work =  [ [1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0] ]

# inputs = np.array([
#     [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ],
#     [ [1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0] ]
# ])
#
# tf.reset_default_graph()
# tf.set_random_seed(777)
# tf_inputs = tf.constant(inputs, dtype=tf.float32)
# rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=3)
# outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, dtype=tf.float32, inputs=tf_inputs)
# variables_names = [v.name for v in tf.trainable_variables()]
#
# print(outputs)
# print(state)
# print("weights")
# for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
#     print(v)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     output_run, state_run = sess.run([outputs, state])
#     print("output values")
#     print(output_run)
#     print("\nstate value")
#     print(state_run)
#     print("weights")
#     values = sess.run(variables_names)
#     for k, v in zip(variables_names, values):
#         print(k, v)