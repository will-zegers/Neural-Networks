%%%%%%%%%%%%%%%
% Import data %
%%%%%%%%%%%%%%%
S1 = importdata('./iris/iris_train.data');
S2  = importdata('./iris/iris_test.data');

for i=1:length(S1.data)
	S1.data(i,5) = strcmp(S1.textdata{i},"Iris-versicolor");
end

for i=1:length(S2.data)
	S2.data(i,5) = strcmp(S2.textdata{i},"Iris-versicolor");
end

traindata = S1.data';
%train_t   = strcmp(S1.textdata,"Iris-versicolor");
testdata  = S2.data';
%test_t   = strcmp(S2.textdata,"Iris-versicolor");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Z-scores for test and training data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu_mat    = ones(4,length(traindata));
sigma_mat = ones(4,length(traindata));

for i=1:4
	mu_mat(i,:)    *= mean(traindata(i,:)); 
	sigma_mat(i,:) *= std(traindata(i,:)); 
end

z_score_train = (traindata(1:4,:) - mu_mat) ./ sigma_mat;
z_score_test  = (testdata(1:4,:)  - mu_mat(:,1:length(testdata))) ./ sigma_mat(:,1:length(testdata));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot each of the 2 dimensional feature spaces %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(3,2,1)
scatter(traindata(1,:),traindata(2,:),10,traindata(5,:),'o','filled')

subplot(3,2,2)
scatter(traindata(1,:),traindata(3,:),10,traindata(5,:),'o','filled')

subplot(3,2,3)
scatter(traindata(1,:),traindata(4,:),10,traindata(5,:),'o','filled')

subplot(3,2,4)
scatter(traindata(2,:),traindata(3,:),10,traindata(5,:),'o','filled')

subplot(3,2,5)
scatter(traindata(2,:),traindata(4,:),10,traindata(5,:),'o','filled')

subplot(3,2,6)
scatter(traindata(2,:),traindata(4,:),10,traindata(5,:),'o','filled')
%%%%%%%%%%%%%%%%%%%%
% Train Perceptron %
%%%%%%%%%%%%%%%%%%%%
w      = zeros(4,1);
thresh = 0;

traindata = traindata(:,randperm(length(traindata)));
for i=1:length(traindata)
	y = (w'*traindata(1:4,i) >= thresh);
	if y ~= traindata(5,i)
		if traindata(5,i) == 1
			w += traindata(1:4,i);
			thresh--;
		else
			w -= traindata(1:4,i);
			thresh++;
		end
	end	
end

test_err= all((w'*testdata(1:4,:) >= 0) ~= testdata(5,:))/70;
fprintf("Test error is %i%%\r\n", test_err)

