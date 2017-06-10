function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);
sum = sum(-y'*log(h) - (1 - y)'*log(1-h));
sum_theta = (theta' .* [0; ones(size(theta, 1) - 1, 1)]') * theta;

%regularized logistic regression
J = (1 / m) * sum + (lambda / (2*m)) * sum_theta;

filtered_theta = [0; theta(2:end)];

%regularized gradient descent for logistic regression
grad = (1 / m) * (X'*(h-y)) + lambda / m * filtered_theta;
%grad(1, : ) = grad(1, : ) - (lambda / m) * theta(1, : );



% =============================================================

end
