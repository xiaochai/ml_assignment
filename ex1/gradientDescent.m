function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %sum = 0;
    %for k = 1:m,
    %    sum = sum + (theta(1, 1)*X(m, 1) + theta(2, 1) *X(m,2) - y(m, 1)) * X(m,1);
    %end
    %theta1 = theta(1,1) - alpha * sum;

    %sum = 0;
    %for k = 1:m,
    %    sum = sum + (theta(1, 1)*X(m, 1) + theta(2, 1) *X(m,2) - y(m, 1)) * X(m,2);
    %end
    %theta2 = theta(2,1) - alpha * sum;

    %theta = [theta1; theta2];
    %newtheta = zeros(length(theta), 1);
    %for k = 1:length(theta),
    %    newtheta(k, 1) = theta(k, 1) - alpha * (X * theta - y)'*X(:, k);
    %end
    %theta = newtheta;

    %theta = theta - alpha/m*(X'*X*theta - X'*y);

     theta = theta - alpha /m * ((X * theta - y)'*X)';

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
