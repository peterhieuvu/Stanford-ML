function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m, 1) X]; %add column of ones to get a1 with examples in rows

z2 = a1 * Theta1';   %calculate z2
a2 = [ones(m, 1) sigmoid(z2)]; %calculate a2 by taking the sigmoid of z2 and adding column of 1s

z3 = a2 * Theta2';	  %calculate z3
output = sigmoid(z3); %h(x)

[useless p] = max(output, [], 2); %get index of max of each row

% =========================================================================


end
