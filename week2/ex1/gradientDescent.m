function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp1 = 0;
temp2 = 0;
cost = 0;
predicted = 0;

for iter = 1:num_iters
    cost = computeCost(X, y, theta);
    predicted = X * theta;
    error = predicted - y;

    % Save the cost J in every iteration
    J_history(iter) = cost;

    % store new thetas in temps
    temp1 = theta(1) - alpha / m * sum(error);
    temp2 = theta(2) - alpha / m * sum(error' * X(:,2));

    % update theta
    theta = [temp1; temp2];



    % ============================================================

end

end
