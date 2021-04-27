clear all
close all

seeds = [42, 69, 322, 1337, 9000];

videos_path = [pwd filesep 'videos' filesep];
frames_path = [pwd filesep 'frames' filesep];
features_path = [pwd filesep 'features' filesep];
networks_path = [pwd filesep 'networks' filesep];
results_path = [pwd filesep 'results' filesep];

if ~exist(videos_path,'dir')
    videos_path = strrep([pwd filesep 'videos' filesep],'sivp','nepl');
end
if ~exist(frames_path,'dir')
    frames_path = strrep([pwd filesep 'frames' filesep],'sivp','nepl');
end
if ~exist(features_path,'dir')
    features_path = strrep([pwd filesep 'features' filesep],'sivp','nepl');
end
if ~exist(networks_path,'dir')
    networks_path = strrep([pwd filesep 'networks' filesep],'sivp','nepl');
end

ft_types = {'correct'; 'faulty'};
test_types = {'independent'; 'tainted'};

% Parameters of the algorithm
Constants.PoolMethod          = 'avg';         % 'max','min','avg', or 'median' can be choosen
Constants.numberOfVideos      = 1200;          % number of videos in the database
Constants.numberOfTrainVideos = 720;           % number of training videos
Constants.numberOfValidationVideos = 240;      % number of validation videos
Constants.path                = path;          % path to videos
Constants.framefrac           = 0.2;

% Parameters for transfer learning
LSTMParameters.trainingOptions    = 'sgdm';        % adam
LSTMParameters.initialLearnRate   = 1e-4;          % 1e-5 
LSTMParameters.miniBatchSize      = 64;            % 32
LSTMParameters.maxEpochs          = 10;            % 100
LSTMParameters.verbose            = false;
LSTMParameters.shuffle            = 'every-epoch';
LSTMParameters.validationPatience = Inf;
LSTMParameters.bgDispatch         = true;
LSTMParameters.checkpointPath     = 'Checkpoints';
LSTMParameters.numClasses         = 5;
LSTMParameters.N                  = 3;    

load KoNViD1k.mat % video names and MOS
Name = [Name{:}]';

if ~exist(videos_path,'dir')
    disp(' Video path not found. Please provide it in line 6.');
    return
end

if ~exist(frames_path,'dir')
    disp(' - Extracting Frames from Videos -- ');
    mkdir(frames_path);
    extractImages(Name, MOS, videos_path, frames_path);
else
    disp(' - Found Frame Data Folder -- ');
end

for ft_type_num = 1:len(ft_types)
    ft_type = ft_types{ft_type_num};
    disp([' - Running ' ft_type ' Fine-Tuning - ']);

    for seednum =1:len(seeds)
        disp(' -- Setting up Image Datastore -- ');
        idsmap = containers.Map;
        for i=1:len(Name)
            idsmap(Name(i)) = imageDatastore(strcat(frames_path, filesep, Name(i)), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        end

        baseseed = seeds(seednum);
        Constants.Seed = baseseed;
        rng(Constants.Seed);
        
        disp([' -- Commencing Fine-Tuning using Seed ' num2str(Constants.Seed) ' --'])

        
        if exist([features_path 'konvid1k_iv3_feats_' ft_type '_ft_' num2str(Constants.Seed) '.mat'],'file')
            disp(' --- Loading Activations --- ');
            % Load the activations
            load([features_path 'konvid1k_iv3_feats_' ft_type '_ft_' num2str(Constants.Seed) '.mat'],'actsstruct');
            load([networks_path 'konvid1k_iv3_trained_network_' ft_type '_ft_' num2str(Constants.Seed) '.mat'],'PermutedName','PermutedMOS','trainDS','valDS','TrainVideos','ValidationVideos','TestVideos');
        else
            if exist([networks_path 'konvid1k_iv3_trained_network_' ft_type '_ft_' num2str(Constants.Seed) '.mat'],'file')
                disp(' --- Loading Fine-Tuned Network --- ');
                % Load fine-tuned IV3
                load([networks_path 'konvid1k_iv3_trained_network_' ft_type '_ft_' num2str(Constants.Seed) '.mat'],'net','PermutedName','PermutedMOS','trainDS','valDS','TrainVideos','ValidationVideos','TestVideos');
            else
                disp(' --- Fine-Tuning IV3 Network --- ');
                % Load pretrained IV3
                net = inceptionv3;

                % Transfer learning
                if(Constants.useTransferLearning)
                    
                    p = randperm(Constants.numberOfVideos);

                    PermutedName = Name(p);    % random permutation of the videos
                    PermutedMOS = MOS(p);      % random permutation of the videos
                    
                    if strcmp(ft_type,'correct')
                        TrainVideos = PermutedName(1:Constants.numberOfTrainVideos);
                        ValidationVideos = PermutedName(Constants.numberOfTrainVideos+1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);
                        TestVideos = PermutedName(Constants.numberOfTrainVideos+Constants.numberOfValidationVideos+1:end);
                        
                        trainDS = filterFrames(idsmap,TrainVideos,Constants.framefrac);
                        valDS = filterFrames(idsmap,ValidationVideos,Constants.framefrac);
                    else
                        TrainVideos = PermutedName(1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);
                        ValidationVideos = PermutedName(1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);
                        TestVideos = PermutedName(Constants.numberOfTrainVideos+Constants.numberOfValidationVideos+1:end);
                        
                        tmpDS = filterFrames(idsmap,TrainVideos,Constants.framefrac);
                        
                        f = tmpDS.Files;
                        l = tmpDS.Labels;
                        tmpp = randperm(len(tmpDS.Files));
                        
                        trainDS = tmpDS.copy();
                        valDS = tmpDS.copy();
                        
                        trainDS.Files = f(tmpp(1:round(len(tmpp)*2/3)));
                        trainDS.Labels = l(tmpp(1:round(len(tmpp)*2/3)));

                        valDS.Files = f(tmpp(round(len(tmpp)*2/3)+1:end));
                        valDS.Labels = l(tmpp(round(len(tmpp)*2/3)+1:end));
                    end
                    numClasses = LSTMParameters.numClasses;

                    lgraph = layerGraph(net);

                    lgraph = removeLayers(lgraph, {'predictions_softmax','ClassificationLayer_predictions'});
                    newLayers = [
                        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
                        softmaxLayer('Name','softmax')
                        classificationLayer('Name','classoutput')];

                    lgraph = addLayers(lgraph, newLayers);
                    lgraph = connectLayers(lgraph,'predictions','fc');

                    
                    if strcmp(ft_type,'correct')
                        numIterationsPerEpoch = floor((numel(trainDS.Labels)/LSTMParameters.miniBatchSize)/20);
                    else
                        numIterationsPerEpoch = floor(numel(trainDS.Labels)/LSTMParameters.miniBatchSize);
                    end
                    
                    options = trainingOptions(LSTMParameters.trainingOptions,...
                        'MiniBatchSize',LSTMParameters.miniBatchSize,...
                        'MaxEpochs',LSTMParameters.maxEpochs,...
                        'InitialLearnRate',LSTMParameters.initialLearnRate,...
                        'Verbose',LSTMParameters.verbose,...
                        'Plots','training-progress',...
                        'ValidationData',valDS,...
                        'ValidationFrequency',numIterationsPerEpoch,...
                        'ValidationPatience',LSTMParameters.validationPatience,...
                        'Shuffle', LSTMParameters.shuffle,...
                        'ExecutionEnvironment','gpu',...
                        'DispatchInBackground',LSTMParameters.bgDispatch,...
                        'CheckPointPath',LSTMParameters.checkpointPath,...
                        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,LSTMParameters.N));

                    net = trainNetwork(trainDS, lgraph, options);
                    save([networks_path 'konvid1k_iv3_trained_network_' ft_type '_ft_' num2str(Constants.Seed) '.mat'],'net','PermutedName','PermutedMOS','trainDS','valDS','TrainVideos','ValidationVideos','TestVideos');
                end
            end
            
            disp(' --- Extracting Activations --- ');
            % Extract the activations
            actsstruct = extractVideoFeatures(idsmap, Name, net);
            save([features_path 'konvid1k_iv3_feats_' ft_type '_ft_' num2str(Constants.Seed) '.mat'],'actsstruct','-v7.3','-nocompression');
        end
        
        disp(' -- Fine-tuning complete. -- ');
        disp(' -- Commencing LSTM Training -- ');
        
        for test_type_num = 1:len(test_types)
            test_type = test_types{test_type_num};
            
            if strcmp(test_type,'independent')
                disp([' --- Training and Evaluating SVR with ' test_type ' Test Set --- ']);
                
                if exist([results_path 'konvid1k_iv3_results_' ft_type '_ft_' test_type '_tests_' num2str(baseseed) '.mat'],'file')
                    load([results_path 'konvid1k_iv3_results_' ft_type '_ft_' test_type '_tests_' num2str(baseseed) '.mat'])
                else
                    [netLSTM, info, PLCC, SROCC, YPRED, YTEST] = trainlstm(TrainVideos, ValidationVideos, TestVideos, Name, MOS, actsstruct, LSTMParameters);

                    save([results_path 'konvid1k_iv3_results_' ft_type '_ft_' test_type '_tests_' num2str(baseseed) '.mat'],'PLCC','SROCC','YPRED','YTEST','TrainVideos','ValidationVideos','TestVideos'); 
                end
                
                disp([' --- Results for ' test_type ' Test Set: ' num2str(round(SROCC,3)) ' SROCC --- '])
            else
                disp([' --- Training and Evaluating 5 SVRs with ' test_type ' Test Sets --- ']);

                num_svrs = 5;

                if exist([results_path 'konvid1k_iv3_results_' ft_type '_ft_' test_type '_tests_' num2str(baseseed) '.mat'],'file')
                    load([results_path 'konvid1k_iv3_results_' ft_type '_ft_' test_type '_tests_' num2str(baseseed) '.mat'])
                else
                    PLCC = cell(num_svrs,1);
                    SROCC = cell(num_svrs,1);
                    KROCC = cell(num_svrs,1);
                    YPRED = cell(num_svrs,1);
                    YTEST = cell(num_svrs,1);
                    
                    for j=1:num_svrs
                        
                        rng(Constants.Seed+j);

                        p = randperm(Constants.numberOfVideos);

                        PermutedName = Name(p);
                        PermutedMOS = MOS(p);

                        TrainVideos = PermutedName(1:Constants.numberOfTrainVideos);
                        ValidationVideos = PermutedName(Constants.numberOfTrainVideos+1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);
                        TestVideos = PermutedName(Constants.numberOfTrainVideos+Constants.numberOfValidationVideos+1:end);

                        [netLSTM, info, PLCC{j}, SROCC{j}, YPRED{j}, YTEST{j}] = trainlstm(TrainVideos, ValidationVideos, TestVideos, Name, MOS, actsstruct, LSTMParameters);
                    end
                    
                    save([results_path 'konvid1k_iv3_results_' ft_type '_ft_' test_type '_tests_' num2str(baseseed) '.mat'],'PLCC','SROCC','YPRED','YTEST','TrainVideos','ValidationVideos','TestVideos'); 
                end
                
                res = zeros(num_svrs,1);
                for j=1:num_svrs
                    res(j) = SROCC{j};
                end

                disp([' --- Results for ' num2str(num_svrs) ' ' test_type ' Test Sets: ' num2str(round(mean(res),3)) ' SROCC --- '])
                
            end
        end
    end
end