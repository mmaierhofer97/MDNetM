function [ result ] = mdnet_run(images, region, net, display, pathSave)
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
img2=imcrop(img,region(1,:));
%% Initialize displayots
colormap = ['y','m','c','r','g','b','w','k'];
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    for m = 1:M
        rectangle('Position', targetLoc(m,:), 'EdgeColor', colormap(m), 'Linewidth', 1);
    end
    set(gca,'position',[0 0 1 1]);
    
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
end

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

for m = 1:M
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
    feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
    pos_data = feat_conv(:,:,:,pos_idx);
    neg_data = feat_conv(:,:,:,neg_idx);


    %% Learning CNN
    fprintf('  training cnn...\n');
    net_fc{m} = mdnet_finetune_hnm(net_fc_init,pos_data,neg_data,opts,...
        'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);

end

%% Initialize displayots
colormap = ['y','m','c','r','g','b','w','k'];
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    for m = 1:M
        rectangle('Position', targetLoc(m,:), 'EdgeColor', colormap(m), 'Linewidth', 1);
    end
    set(gca,'position',[0 0 1 1]);
    
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
end

%% Prepare training data for online update
total_pos_data = cell(1,1,1,M,nFrames);
total_neg_data = cell(1,1,1,M,nFrames);
for m=1:M
    neg_examples = gen_samples('uniform', targetLoc(m,:), opts.nNeg_update*2, opts, 2, 5);
    r = overlap_ratio(neg_examples,targetLoc(m,:));
    neg_examples = neg_examples(r<opts.negThr_init,:);
    neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

    examples = [pos_examples; neg_examples];
    pos_idx = 1:size(pos_examples,1);
    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

    feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
    total_pos_data{m,1} = feat_conv(:,:,:,pos_idx);
    total_neg_data{m,1} = feat_conv(:,:,:,neg_idx);

    success_frames{m} = [1];
    trans_f(m) = opts.trans_f;
    scale_f(m) = opts.scale_f;
end

%%%%%%%%%%%%%%
occBool = repmat({true}, M, 1);
occFrames = cell(M,1);
occSamples = cell(M,1);
occImg = cell(M,1);
occStart = cell(M,1);
occEnd = cell(M,1);
nSamp=repmat({5}, M, 1);
occSim=cell(M,1);
%%%%%%%%%%%%%%

%% Main loop
for To = 2:nFrames;
    fprintf('Processing frame %d/%d... ', To, nFrames)  ;
    
    img = imread(images{To});
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    spf = tic;
    %% Estimation
    
    for m = 1:M
        % draw target candidates
        samples = gen_samples('gaussian', targetLoc(m,:), opts.nSamples, opts, trans_f(m), scale_f(m));
        feat_conv = mdnet_features_convX(net_conv, img, samples, opts);
        % evaluate the candidates
        feat_fc = mdnet_features_fcX(net_fc{m}, feat_conv, opts);
        feat_fc = squeeze(feat_fc)';
        [scores,idx] = sort(feat_fc(:,2),'descend');
        target_score = mean(scores(1:nSamp{m}));
        targetLoc(m,:) = round(mean(samples(idx(1:nSamp{m}),:)));
        % final target
        result(m,To,:) = targetLoc(m,:);

        % extend search space in case of failure
        if(target_score<0)
            trans_f(m) = min(1.5, 1.1*trans_f(m));
        else
            trans_f(m) = opts.trans_f;
        end
        
        %%%%%%%%%%%%%%%%%%%%%% Occlusion Interpolation %%%%%%%%%%%%%%%%%%%
        target_score
        if (target_score<0)
            if (occBool{m}==true)
                nSamp{m}=2;
                img2 = imread(images{To-1});
                %fconv = mdnet_features_convX(net_conv, img2, result(m,To-1,:), opts);
                %occStart{m} = fconv(:,:,:,1);
                occSamples{m}{end+1}=squeeze(result(m,To-1,:));
                occImg{m}{end+1}=img2;
                occBool{m} = false;
            end
            occFrames{m}{end+1}=feat_conv;
            occSamples{m}{end+1}=samples;
            occImg{m}{end+1}=img;
        else
            if (occBool{m}==false)
                nSamp{m}=5;
                occBool{m} = true;
                
                %occEnd{m} = feat_conv(:,:,:,idx(1));
                occSamples{m}{end+1}=squeeze(result(m,To,:));
                occImg{m}{end+1}=img;

                [sim,ind] = SimDP_ch(occImg{m},occSamples{m});
                %[sim,ind] = SimDP(occStart{m},occFrames{m},occEnd{m});
                ind;
                for x=1:length(ind)
                    occSim{m}(:,end+1) = [To-(length(ind)+1)+x; sim{x}(ind(x))];
                end
                fileID = fopen([num2str(m) 'exp.txt'],'w');
                fprintf(fileID,'%6s %12s\n','x','sim');
                fprintf(fileID,'%6.2f %12.8f\n',occSim{m});
                fclose(fileID);
               
                for i=2:length(ind)
                    frI = To-(length(ind)+1)+i
                    sim{i}(ind(i));
                    if (sim{i}(ind(i)))>0.8
                        result(m,frI,:) = occSamples{m}{i}(ind(i),:);
                        %result(m,frI,:) = occSamples{m}{i-1}(ind(i),:);
                         h=''
                    else
                        result(m,frI,:) = result(m,frI-1,:);
                        h='h'
                    end
                    img = imread(images{frI});
                    if display
                        hc = get(gca, 'Children'); delete(hc(1:end-1));
                        set(hd,'cdata',img); hold on;

                        for c = 1:M
                            rectangle('Position', result(c,frI,:), 'EdgeColor', colormap(c), 'Linewidth', 1);
                        end

                        %%rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
                        set(gca,'position',[0 0 1 1]);

                        text(10,10,[num2str(frI) h],'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
                        hold off;
                        imwrite(frame2im(getframe(gcf)), [ pathSave 'b' num2str(frI) '.jpg']);

                        drawnow;
                    end
                    
                end
                img = imread(images{To});
                occFrames{m} = cell(0);
                occSamples{m} = cell(0);
                occImg{m} = cell(0);
            end
        end
                    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % bbox regression
        if(opts.bbreg && target_score>0)
            X_ = permute(gather(feat_conv(:,:,:,idx(1:5))),[4,3,1,2]);
            X_ = X_(:,:);
            bbox_ = samples(idx(1:5),:);
            pred_boxes = predict_bbox_regressor(bbox_reg{m}.model, X_, bbox_);
            result(m,To,:) = round(mean(pred_boxes,1));
        end

        %% Prepare training data
        if(target_score>0)
            pos_examples = gen_samples('gaussian', targetLoc(m,:), opts.nPos_update*2, opts, 0.1, 5);
            r = overlap_ratio(pos_examples,targetLoc(m,:));
            pos_examples = pos_examples(r>opts.posThr_update,:);
            pos_examples = pos_examples(randsample(end,min(opts.nPos_update,end)),:);

            neg_examples = gen_samples('uniform', targetLoc(m,:), opts.nNeg_update*2, opts, 2, 5);
            r = overlap_ratio(neg_examples,targetLoc(m,:));
            neg_examples = neg_examples(r<opts.negThr_update,:);
            neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

            examples = [pos_examples; neg_examples];
            pos_idx = 1:size(pos_examples,1);
            neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

            feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
            total_pos_data{m,To} = feat_conv(:,:,:,pos_idx);
            total_neg_data{m,To} = feat_conv(:,:,:,neg_idx);

            success_frames{m} = [success_frames{m}, To];
            if(numel(success_frames{m})>opts.nFrames_long)
                total_pos_data{m,success_frames{m}(end-opts.nFrames_long)} = single([]);
            end
            if(numel(success_frames{m})>opts.nFrames_short)
                total_neg_data{m,success_frames{m}(end-opts.nFrames_short)} = single([]);
            end
        else
            total_pos_data{m,To} = single([]);
            total_neg_data{m,To} = single([]);
        end

        %% Network update

        if((mod(To,opts.update_interval)==0 || target_score<0) && To~=nFrames)
            
            if (target_score<0) % short-term update
                temp_data = cell(1,1,1,min(opts.nFrames_short, size(success_frames{m},2)));
                temp_data(1,1,:,:) = total_pos_data(m,success_frames{m}(max(1,end-opts.nFrames_short+1):end));
                pos_data = cell2mat(temp_data);
            else % long-term update
                temp_data = cell(1,1,1,min(opts.nFrames_long, size(success_frames{m},2)));
                temp_data(1,1,:,:) = total_pos_data(m,success_frames{m}(max(1,end-opts.nFrames_long+1):end));
                pos_data = cell2mat(temp_data);
            end
            temp_data = cell(1,1,1,min(opts.nFrames_short, size(success_frames{m},2)));
            temp_data(1,1,:,:) = total_neg_data(m,success_frames{m}(max(1,end-opts.nFrames_short+1):end));
            neg_data = cell2mat(temp_data);

    %         fprintf('\n');
            net_fc{m} = mdnet_finetune_hnm(net_fc{m},pos_data,neg_data,opts,...
                'maxiter',opts.maxiter_update,'learningRate',opts.learningRate_update);
        end

        
    end
    
    spf = toc(spf);
    fprintf('%f seconds\n',spf);
    %% Display
    
    if display 
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',img); hold on;
        
        for m = 1:M
            rectangle('Position', result(m,To,:), 'EdgeColor', colormap(m), 'Linewidth', 1);
        end
        
        %%rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
        set(gca,'position',[0 0 1 1]);
        
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        imwrite(frame2im(getframe(gcf)), [pathSave num2str(To) '.jpg']);
       
        drawnow;
    end
end