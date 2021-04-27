function [netLSTM, info, PLCC, SROCC, YPred, Y_test] = trainlstm(TrainVideos, ValidationVideos, TestVideos, Name, MOS, actsstruct, ParametersTransferLearning)
    
    sizes = cellfun(@(x) (size(x,2)),actsstruct.Features_full);
    
    X = actsstruct.Features_full;
    t = table(MOS, X, sizes, 'RowNames', Name);
    
    Train = sortrows(t([TrainVideos; ValidationVideos],:),'sizes','descend');
    Test = sortrows(t(TestVideos,:),'sizes','descend');
    X_train = Train.X;
    Y_train = Train.MOS;
    X_test = Test.X;
    Y_test = Test.MOS;

    clear t;
    clear actsstruct;
    
    layers = [ ...
        sequenceInputLayer(2048)
        lstmLayer(1024,'OutputMode', 'last')
        lstmLayer(128,'OutputMode', 'last')
        fullyConnectedLayer(1)
        regressionLayer];
    options = trainingOptions('adam',...
        'MiniBatchSize',27,...
        'MaxEpochs',ParametersTransferLearning.maxEpochs,...
        'InitialLearnRate',ParametersTransferLearning.initialLearnRate,...
        'Verbose',ParametersTransferLearning.verbose,...
        'Plots','training-progress',...
        'GradientThreshold', 0.5,...
        'ExecutionEnvironment','gpu',...
        'DispatchInBackground',ParametersTransferLearning.bgDispatch);

    [netLSTM, info] = trainNetwork(X_train,Y_train, layers, options);
    
    close all;
    
    YPred = predict(netLSTM, X_test);
    
    PLCC = corr(YPred, Y_test, 'Type', 'Pearson');
    SROCC = corr(YPred, Y_test, 'Type', 'Spearman');
    
end