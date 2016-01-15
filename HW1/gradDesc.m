function[x] = getTestSize()
    x = 10000;
end

function[x] = getTrainSize()
    x = 20000;
end 

%%%
% Reads a file, and returns a matrix shaped to represent the data as vectors
% in d-dimensional space
%%%
function[mat] = readFile(fileName,offset,d,sample_size)
	
	mat = [];

	fd = fopen(fileName);
	fread(fd, offset, 'uint8');
	A = fread(fd, d*sample_size, 'uint8');
	mat = reshape(A,[d,sample_size]);
    fclose(fd);
end

%%%
% Wrapper function to read images and return a matrix of vectorized pixel
% values.
%%%
function[mat] = readImages(fileName,sample_size)
	mat = readFile(fileName,16,28*28,sample_size);
	mat = [ones(1, size(mat)(2) );mat];
end

%%%
% Wrapper function to read the 'teacher' labels of the images, returning a
% 1 x sample_size vector.
%%%
function[mat] = readLabels(fileName,sample_size)
	mat = readFile(fileName,8,1,sample_size);
end

%%%
% Encodes a matrix of decimal and returns a matrix of one-hot mappings.
%%%
function[y] = toOneHot(v)
	m = length(v);

	y = zeros(m,10);
	for i=1:m
		y(i,v(i)+1) = 1; 
	end 
end

%%%
% Takes a matrix of either one-hot values or probabiliy values, and returns the
% decimal numbers corresponding to the numbers in the selected position (minus,
% one, since MATLab is 1-indexed.
%%%
function[y] = fromOneHot(v)
    [a,y] = max(v');
	y -= 1; 
end

%%%
% Takes a matrix of weights and data values, and returns the value of the 
% squash function.
%%%
function[y] = sigmoid(w,x)
	y = 1 ./ (1 + exp(w'*x) )';
end

%%%
% Runs stochastic gradient descent on given data, teacher, weights, and a
% step parameter.
%%%
function[w] = gradientDescent(x,t,w,step)
	m = size(x)(2);
	
	for i=1:m
		w -= step * bsxfun('times', (t(i,:) - sigmoid(w,x(:,i) ) ), x(:,i));	
	end
end

%%%
% Trains 2-mode classification using linear regression
%%%
function[W] = linearRegTrain(x,t,alpha,iter)
    W = [];
    for digit=0:9
        w = zeros(785,1);

        digitClassif = (t == digit)';
        for i=i:iter
            w = gradientDescent(x,digitClassif,w,alpha);
        end
        W = [W w];
    end
end

%%%
% Trains 10-mode classification using soft max
%%%
function[W] = softMaxTrain(x,t,alpha,iter)
	t = toOneHot(t);
    W = zeros(785,10);

	for i=1:iter
    	W = gradientDescent(x, t, W, alpha);
	end
end

%%%
% Runs soft max training and then computes the testing accuracy of the learned
% weights. (Note: the dummy return value is just a place holder, so the softMax
% and linear regression function can be interchangeable in the top-level
% function.
%%%
function[overallAccuracy, dummy] = softMax(x_train, t_train, x_test, t_test, alpha, iter)
    W = softMaxTrain(x_train, t_train, alpha, iter);
    y = fromOneHot(sigmoid(W,x_test) );
    overallAccuracy = sum(y == t_test) / getTestSize();
end

%%%
% Runs linear regression training and then computes the testing accuracy of
% the learned weights.
%%%
function [overallAccuracy, digitAccuracy] = linearRegression(x_train, t_train, x_test, t_test, alpha, iter)
    W = linearRegTrain(x_train, t_train, alpha, iter);
    y = sigmoid(W,x_test);
    digitAccuracy = ( sum((y >= 0.5) == toOneHot(t_test)) ) / getTestSize();
    overallAccuracy = sum(fromOneHot(y) == t_test) / getTestSize();
end

%%%
% Top level function to run gradient descent learning using the given function
% (either 'softMax' or 'linearRegression').
%%%
function[] = runGradientDescent(method,alpha,iter)
	images = {
	    './train-images-idx3-ubyte',
	    './t10k-images-idx3-ubyte',
	    };
	
	labels = {
	    './train-labels-idx1-ubyte',
	    './t10k-labels-idx1-ubyte',
	    };
	
	x_train = readImages(images{1}, getTrainSize() );
	t_train = readLabels(labels{1}, getTrainSize() );
	x_test  = readImages(images{2}, getTestSize() );
	t_test  = readLabels(labels{2}, getTestSize() );

    [overall,digit] = feval(method,x_train,t_train,x_test,t_test,alpha,iter)
end
