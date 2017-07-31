function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% add ones column vector to X
X = [ones(size(X, 1), 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% The hidden layer or layer_2
layer_2 = sigmoid(Theta1 * X');

% add ones column vector to layer_2
layer_2 = [ones(size(layer_2, 2), 1)'; layer_2];

% the output layer or layer_3 also contains our hypothesis values
layer_3 = layer_2' * Theta2';

[temp, p] = max(layer_3, [], 2);
end
