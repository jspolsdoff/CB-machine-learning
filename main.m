% Cyclebar Machine Learning Script
%
% using multivariant gradient descent predict cycle points

%% Clear and Close Figures
clear ; close all; clc

% Load Data
file_name = input("Enter file name for analysis: ", "s");
fprintf('Loading data ...\n');
data = load("cycle_stats.txt");
y_col = columns(data);
num_features = y_col - 1;
X = data(:, 1:num_features);
y = data(:, y_col);
m = length(y);

% TEST: check to see if data was loaded correctly
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

% having trouble with normalize with gender feature since causes division by 0
% [X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Choose some alpha value
alpha = 0.01; 
num_iters = 400; 

% Init Theta and Run Gradient Descent 
theta = zeros(num_features + 1, 1);

% TEST: print out what theta is before
fprintf(' %f \n', theta);

[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% TEST: print out what theta is after
fprintf(' %f \n', theta);

% Plot the convergence graph
%figure;
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');

% Display gradient descent's result

% Solve using normal equation
fprintf('Solving with normal equations...\n');

% Load Data
fprintf('Loading data ...\n');
data = load("cycle_stats.txt");
y_col = columns(data);
num_features = y_col - 1;
X = data(:, 1:num_features);
y = data(:, y_col);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% ============== TESTING ZONE ======================================
cycle_points = [1 255 78 235 44 1]*theta; % 

fprintf(['Predicted Cycle Points for your stats ' ...
         '(using normal equations):\n %f\n'], fix(cycle_points));



