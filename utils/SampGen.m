function [ pos_data,neg_data ] = SampGen(images, region, net, display, pathSave)
% MDNET_RUN
% Main interface for MDNet tracker
%
% INPUT:
%   images  - 1xN cell of the paths to image sequences
%   region  - 1x4 vector of the initial bounding box [left,top,width,height]
%   net     - The path to a trained MDNet
%   display - True for displying the tracking result
%
% OUTPUT:
%   result - Nx4 matrix of the tracking result Nx[left,top,width,height]
%
% Hyeonseob Nam, 2015
% 

if(nargin<4), display = true; end

%% Initialization
fprintf('Initialization...\n');

nFrames = length(images);
M = size(region, 1);

img = imread(images{1});
if(size(img,3)==1), img = cat(3,img,img,img); end


targetLoc = region;
result = zeros(M, nFrames, 4); result(:,1,:) = targetLoc;

[net_conv, net_fc_init, opts] = mdnet_init(img, net);

%% Train a bbox regressor
if(opts.bbreg)
    for m = 1:M
        pos_examples = gen_samples('uniform_aspect', targetLoc(m,:), opts.bbreg_nSamples*10, opts, 0.3, 10);
        r = overlap_ratio(pos_examples,targetLoc(m,:));
        pos_examples = pos_examples(r>0.6,:);
        pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
        feat_conv = mdnet_features_convX(net_conv, img, pos_examples, opts);

        X = permute(gather(feat_conv),[4,3,1,2]);
        X = X(:,:);
        bbox = pos_examples;
        bbox_gt = repmat(targetLoc(m,:),size(pos_examples,1),1);
        bbox_reg{m} = train_bbox_regressor(X, bbox, bbox_gt);
    end
end

%% Extract training examples
fprintf('  extract features...\n');

for m = 1:1
    % draw positive/negative samples
    pos_examples = gen_samples('gaussian', targetLoc(m,:), opts.nPos_init*2, opts, 0.1, 5);
    r = overlap_ratio(pos_examples,targetLoc(m,:));
    pos_examples = pos_examples(r>opts.posThr_init,:);
    pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

    neg_examples = [gen_samples('uniform', targetLoc(m,:), opts.nNeg_init, opts, 1, 10);...
        gen_samples('whole', targetLoc(m,:), opts.nNeg_init, opts)];
    r = overlap_ratio(neg_examples,targetLoc(m,:));
    neg_examples = neg_examples(r<opts.negThr_init,:);
    neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

    examples = [pos_examples; neg_examples];
    pos_idx = 1:size(pos_examples,1);
    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

    % extract conv3 features
    feat_conv = mdnet_features_convSim(net_conv, img, examples, opts);
    pos_data = feat_conv(:,:,:,pos_idx);
    neg_data = feat_conv(:,:,:,neg_idx);


    %% Learning CNN
    fprintf('  training cnn...\n');
    net_fc{m} = mdnet_finetune_hnm(net_fc_init,pos_data,neg_data,opts,...
        'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);

end
