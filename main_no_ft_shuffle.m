clear all
close all

seeds = [42, 69, 322, 1337, 9000];

videos_path = strrep([pwd filesep 'videos' filesep],'sivp','nepl');
frames_path = strrep([pwd filesep 'frames' filesep],'sivp','nepl');
features_path = strrep([pwd filesep 'features' filesep],'sivp','nepl');
networks_path = strrep([pwd filesep 'networks' filesep],'sivp','nepl');
results_path = [pwd filesep 'results' filesep];

% Parameters of the algorithm
Constants.PoolMethod          = 'avg';         % 'max','min','avg', or 'median' can be choosen
Constants.numberOfVideos      = 1200;          % number of videos in the database
Constants.numberOfTrainVideos = 720;           % number of training videos
Constants.numberOfValidationVideos = 240;      % number of validation videos
Constants.path                = path;          % path to videos
Constants.useSeed             = true;
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
    disp(' Video path not found. Please provide it in line 7.');
    return
end

disp(' -- Setting up Image Datastore -- ');
idsmap = containers.Map;
for i=1:len(Name)
    idsmap(Name(i)) = imageDatastore(strcat(frames_path, filesep, Name(i)), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end

if ~exist(frames_path,'dir')
    disp(' - Extracting Frames from Videos -- ');
    mkdir(frames_path);
    extractImages(Name, MOS, videos_path, frames_path);
else
    disp(' - Found Frame Data Folder -- ');
end

if exist([features_path 'konvid1k_iv3_feats_no_ft.mat'],'file')
    disp(' --- Loading Activations --- ');
    % Load the activations
    load([features_path 'konvid1k_iv3_feats_no_ft.mat'],'actsstruct');
else
    net = inceptionv3;
    disp(' --- Extracting Activations --- ');
    % Extract the activations
    actsstruct = extractVideoFeatures(idsmap, Name, net);
    save([features_path 'konvid1k_iv3_feats_no_ft.mat'],'actsstruct','-v7.3','-nocompression');
end

perf = cell(len(seeds),1);

for seednum =1:len(seeds)

    baseseed = seeds(seednum);
    disp([' -- Commencing SVR Training using Seed ' num2str(baseseed) ' -- ']);

    if exist([results_path 'konvid1k_iv3_results_no_ft_' num2str(baseseed) '.mat'],'file')
        load([results_path 'konvid1k_iv3_results_no_ft_' num2str(baseseed) '.mat'])
    else
        rng(baseseed);
        p = randperm(Constants.numberOfVideos);

        PermutedName = Name(p);
        PermutedMOS = MOS(p);

        TrainVideos = PermutedName(1:Constants.numberOfTrainVideos);
        ValidationVideos = PermutedName(Constants.numberOfTrainVideos+1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);
        TestVideos = PermutedName(Constants.numberOfTrainVideos+Constants.numberOfValidationVideos+1:end);

        [netLSTM, info, PLCC, SROCC, YPRED, YTEST] = trainlstm(TrainVideos, ValidationVideos, TestVideos, Name, MOS, actsstruct, LSTMParameters);
        save([results_path 'konvid1k_iv3_results_no_ft_' num2str(baseseed) '.mat'],'PLCC','SROCC','YPRED','YTEST','TrainVideos','ValidationVideos','TestVideos');
    end
    perf{seednum} = SROCC;
end

res = zeros(len(seeds),1);
for j=1:len(seeds)
    res(j) = perf{j};
end

disp([' --- Results for ' num2str(len(seeds)) ' Test Sets: ' num2str(round(mean(res),3)) ' SROCC --- '])