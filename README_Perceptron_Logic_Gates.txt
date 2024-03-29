Perceptron Logic Gates
In this project, we will use perceptrons to model the fundamental building blocks of computers — logic gates.


For example, the table below shows the results of an AND gate. Given two inputs, an AND gate will output a 1 only if both inputs are a 1:

Input 1	Input 2	Output
0	0	0
0	1	0
1	0	0
1	1	1

We’ll discuss how an AND gate can be thought of as linearly separable data and train a perceptron to perform AND.

We’ll also investigate an XOR gate — a gate that outputs a 1 only if one of the inputs is a 1:

Input 1	Input 2	Output
0	0	0
0	1	1
1	0	1
1	1	0

We’ll think about why an XOR gate isn’t linearly separable and show how a perceptron fails to learn XOR.