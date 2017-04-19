# -*- coding: utf-8 -*-

# the data set is just made for Jeju ML camp
# It is too small to seperate the train and test set
import numpy as np
import pandas as pd
import tensorflow as tf

# using csv data instead of MySQL data
ice_sales = pd.read_csv(
    "icecream_data.csv",
    sep=",",
    header=0)


icecream = np.array(ice_sales, dtype=float)
temperature_data = icecream[:, [0]]
student_data = icecream[:, [1]]
y_data = icecream[:, [2]]

# using one linear regression to get the numbers of students will visit
def student_num(self):
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 1])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([1, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis
    hypothesis = tf.matmul(X, W) + b

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-5)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: temperature_data, Y: student_data})
        # if step % 500 == 0:
        #     print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
    student = sess.run(hypothesis, feed_dict={X: [[self]]})
    return student


# using the one linear regression to get how many icecream will be sold
def ice_qty(self):
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 1])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([1, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis
    hypothesis = tf.matmul(X, W) + b

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-5)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: student_data, Y: y_data})
    sales = sess.run(hypothesis, feed_dict={X: [[self]]})
    return sales

# using the functions to get the expected income
def total_sale(self):
    student = student_num(self)
    student = float(student)
    answer = format(int(student), ',')

    qty = ice_qty(student)

    total_qty = float(qty)
    total_income = total_qty * 1500
    total_income = format(int(round(total_income, -2)), ',')

    print("예상 방문고객은 약 %s명이며, 예상수익은 %s원입니다." % (answer, total_income))
    print("재고수량은 %d개로 맞춰주세요." % total_qty)



