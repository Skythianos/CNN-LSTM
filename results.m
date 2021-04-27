clear all
close all

seeds = [42, 69, 322, 1337, 9000];

results_path = [pwd filesep 'results' filesep];

ft_types = {'no'; 'correct'; 'faulty'};
test_types = {'independent'; 'tainted'};

PLCC_no_ft = zeros(5,1);
SROCC_no_ft = zeros(5,1);

PLCC_correct_ft_independent_test = zeros(5,1);
SROCC_correct_ft_independent_test = zeros(5,1);

PLCC_correct_ft_tainted_test = zeros(5,1);
SROCC_correct_ft_tainted_test = zeros(5,1);

PLCC_faulty_ft_independent_test = zeros(5,1);
SROCC_faulty_ft_independent_test = zeros(5,1);

PLCC_faulty_ft_tainted_test = zeros(5,1);
SROCC_faulty_ft_tainted_test = zeros(5,1);

for ft_type_num = 1:len(ft_types)
    ft_type = ft_types{ft_type_num};
    
    for seednum = 1:len(seeds)
        baseseed = seeds(seednum);
        if strcmp(ft_type,'no')
            load([results_path 'konvid1k_iv3_results_no_ft_' num2str(baseseed) '.mat'],'PLCC','SROCC')
            PLCC_no_ft(seednum) = PLCC;
            SROCC_no_ft(seednum) = SROCC;
        elseif strcmp(ft_type,'correct')
            for testnum = 1:len(test_types)
                test_type = test_types{testnum};
                load([results_path 'konvid1k_iv3_results_' ft_type '_ft_' test_type '_tests_' num2str(baseseed) '.mat'],'PLCC','SROCC')
                if strcmp(test_type,'independent')
                    PLCC_correct_ft_independent_test(seednum) = PLCC;
                    SROCC_correct_ft_independent_test(seednum) = SROCC;
                else
                    tmp_PLCC = zeros(5,1);
                    tmp_SROCC = zeros(5,1);
                    for realization = 1:5
                        tmp_PLCC(realization) = PLCC{realization};
                        tmp_SROCC(realization) = SROCC{realization};
                    end
                    PLCC_correct_ft_tainted_test(seednum) = mean(tmp_PLCC);
                    SROCC_correct_ft_tainted_test(seednum) = mean(tmp_SROCC);
                end
            end
        else
            for testnum = 1:len(test_types)
                test_type = test_types{testnum};
                load([results_path 'konvid1k_iv3_results_' ft_type '_ft_' test_type '_tests_' num2str(baseseed) '.mat'],'PLCC','SROCC')
                if strcmp(test_type,'independent')
                    PLCC_faulty_ft_independent_test(seednum) = PLCC;
                    SROCC_faulty_ft_independent_test(seednum) = SROCC;
                else
                    tmp_PLCC = zeros(5,1);
                    tmp_SROCC = zeros(5,1);
                    for realization = 1:5
                        tmp_PLCC(realization) = PLCC{realization};
                        tmp_SROCC(realization) = SROCC{realization};
                    end
                    PLCC_faulty_ft_tainted_test(seednum) = mean(tmp_PLCC);
                    SROCC_faulty_ft_tainted_test(seednum) = mean(tmp_SROCC);
                end
            end
        end
    end
end

sprintf('%1.2f ($\\pm$%1.2f) & %1.2f ($\\pm$%1.2f)\n%1.2f ($\\pm$%1.2f) & %1.2f ($\\pm$%1.2f)\n%1.2f ($\\pm$%1.2f) & %1.2f ($\\pm$%1.2f)\n%1.2f ($\\pm$%1.2f) & %1.2f ($\\pm$%1.2f)\n%1.2f ($\\pm$%1.2f) & %1.2f ($\\pm$%1.2f)', [mean(PLCC_no_ft),std(PLCC_no_ft),mean(SROCC_no_ft),std(SROCC_no_ft),mean(PLCC_faulty_ft_tainted_test),std(PLCC_faulty_ft_tainted_test),mean(SROCC_faulty_ft_tainted_test),std(SROCC_faulty_ft_tainted_test),mean(PLCC_faulty_ft_independent_test),std(PLCC_faulty_ft_independent_test),mean(SROCC_faulty_ft_independent_test),std(SROCC_faulty_ft_independent_test),mean(PLCC_correct_ft_tainted_test),std(PLCC_correct_ft_tainted_test),mean(SROCC_correct_ft_tainted_test),std(SROCC_correct_ft_tainted_test),mean(PLCC_correct_ft_independent_test),std(PLCC_correct_ft_independent_test),mean(SROCC_correct_ft_independent_test),std(SROCC_correct_ft_independent_test)])