import tensorflow as tf

#hl_nodes is an array of the how many nodes are in a hidden layer
#example: hl1, hl2, hl3 = 500, 500, 500
#hl_nodes = [hl1, hl2, hl3]

# classes is the number of output nodes

# input_size is how many nodes are in the input

# data is the train data

def hl(classes, input_size, hl_nodes, data):
    hidden_layer = [{'weights': tf.Variable(tf.random_normal([input_size, hl_nodes[0]])),
                      'biases': tf.Variable(tf.random_normal([hl_nodes[0]]))}]
    l = [tf.nn.relu(tf.add(tf.matmul(data, hidden_layer[0]['weights']), hidden_layer[0]['biases']))]

    for i in range(len(hl_nodes)):
        hidden_layer.append({'weights': tf.Variable(tf.random_normal([hl_nodes[i-1], hl_nodes[i]])),
                      'biases': tf.Variable(tf.random_normal([hl_nodes[i]]))})

    output_layer = {'weights': tf.Variable(tf.random_normal([hl_nodes[i-1], classes])),
                      'biases': tf.Variable(tf.random_normal([classes]))}

    for n in range(len(hl_nodes)):

        l.append(tf.nn.relu(tf.add(tf.matmul(data, hidden_layer[n]['weights']), hidden_layer[n]['biases'])))

    output = tf.matmul(l[n], output_layer['weights']) + output_layer['biases']

    return hidden_layer, output_layer, l, output

#print(hl(0, 784, [500, 500, 500]))

