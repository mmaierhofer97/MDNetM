function [ result ] = mdnet_run2(images, net, display, pathSave, det)
% % MDNET_RUN
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
% save(['.' conf.seqName 'res'])

if(nargin<4), display = true; end

initLoc = det(det(:,1)==1 & det(:,7)>0,3:6);
%% Initialization
fprintf('Initialization...\n');
nFrames = length(images);
M = size(initLoc, 1)
img = imread(images{1});
if(size(img,3)==1), img = cat(3,img,img,img); end
%% Initialize displayots
colormap = ['y','m','c','r','g','b','w','k'];
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    for m = 1:M
        rectangle('Position', initLoc(m,:), 'EdgeColor', colormap(mod(m,8)+1), 'Linewidth', 1);
        text(initLoc(m,1),initLoc(m,2),num2str(m));
    end
    set(gca,'position',[0 0 1 1]);
    
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
end

targetLoc = initLoc;
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
    %fprintf('  training cnn...\n');
    net_fc{m} = mdnet_finetune_hnm(net_fc_init,pos_data,neg_data,opts,...
        'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);

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
save('16-11-set3')
%load('16-11-set3')
%% Initialize displayots
colormap = ['y','m','c','r','g','b','w','k'];

if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    for m = 1:M
        rectangle('Position', targetLoc(m,:), 'EdgeColor', colormap(mod(m,8)+1), 'Linewidth', 1);
        text(targetLoc(m,1),targetLoc(m,2),num2str(m));
    end
    set(gca,'position',[0 0 1 1]);
    
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
end
%%%%%%%%%%%%%%
occBool = repmat({true}, M, 1);
occFrames = cell(M,1);
occSamples = cell(M,1);
occStart = cell(M,1);
occEnd = cell(M,1);
nSamp=repmat({5}, M, 1);
occCount=cell(M,1);
target_score=cell(M,1);
ovBool=0;
ovConv = cell(M,1);
ovSamp = cell(M,1);
SampStart = cell(M,1);
SampEnd = cell(M,1);
%%%%%%%%%%%%%%
%% Main loop
for To = 2:nFrames
    fprintf('Processing frame %d/%d... ', To, nFrames);
    
    img = imread(images{To});
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    detLoc = det(det(:,1)==To,3:6);
    spf = tic;
    %% Estimation
    
    for m = 1:M
        %check the detection result first
        r = overlap_ratio(detLoc,result(m,To-1,:));
        if occBool{m}==true
            samplesD = detLoc(r>0.7,:);
        else
           samplesD = detLoc(r>0.7,:); 
           %size(detLoc)
        end
        if size(samplesD,1) > 0
            feat_conv = mdnet_features_convX(net_conv, img, samplesD, opts);
            feat_fc = mdnet_features_fcX(net_fc{m}, feat_conv, opts);
            feat_fc = squeeze(feat_fc)';
            [scores,idx] = sort(feat_fc(:,2),'descend');
            target_score{m}= scores(1);
        else
            target_score{m} = -1;
        end
        
        if target_score{m} > 0
            targetLoc(m,:) = samplesD(idx(1),:);
            result(m,To,:) = targetLoc(m,:);
            occFrames{m} = mdnet_features_convX(net_conv, img, targetLoc(m,:), opts);
            occSamples{m} = targetLoc(m,:);
            fprintf('Detection ');
        else
            fprintf('Tracking ');
    % draw target candidates
    %         if success_frames{m}(end) == To-1
    %             r = overlap_ratio(detLoc,result(m,To-1,:));
    %             samples_det = detLoc(r>0,:);
    %         else
    %             samples_det = detLoc;
    %         end
            samples = gen_samples('gaussian', targetLoc(m,:), opts.nSamples, opts, trans_f(m), scale_f(m));
    %         samples = [samples; samples_det];
            feat_conv = mdnet_features_convX(net_conv, img, samples, opts);

            % evaluate the candidates
            feat_fc = mdnet_features_fcX(net_fc{m}, feat_conv, opts);
            feat_fc = squeeze(feat_fc)';
            [scores,idx] = sort(feat_fc(:,2),'descend');
            target_score{m} = mean(scores(1:nSamp{m}));
            targetLoc(m,:) = round(mean(samples(idx(1:nSamp{m}),:)));
            occFrames{m} = cat(4,mdnet_features_convX(net_conv, img, targetLoc(m,:), opts), feat_conv(:,:,:,idx));

            % final target
    %         if target_score{m} > 0
    %             result(m,To,:) = targetLoc(m,:);
    %         else
    %             result(m,To,:) = result(m,To-1,:);
    %         end
            result(m,To,:) = targetLoc(m,:);
            occSamples{m} = [targetLoc(m,:);samples(idx,:)];
            
            


            % bbox regression
            if(opts.bbreg && target_score{m}>0)
                X_ = permute(gather(feat_conv(:,:,:,idx(1:5))),[4,3,1,2]);
                X_ = X_(:,:);
                bbox_ = samples(idx(1:5),:);
                pred_boxes = predict_bbox_regressor(bbox_reg{m}.model, X_, bbox_);
                result(m,To,:) = round(mean(pred_boxes,1));
            end
        end
    target_score{m};
    end
        %%%%%%%%%%%%%%%%%%%%%% Overlap Logic %%%%%%%%%%%%%%%%%%%%%%%%
    for m=1:M-1
        for n=m+1:M
            try ovBool(m,n)==0;
            catch ovBool(m,n)=0;
            end
            try ovBool(n,m)==0;
            catch ovBool(n,m)=0;
            end
            thr=0.4;
            thS=0.6;
            sim = cosine_sim(occFrames{m}(:,:,:,1),occFrames{n}(:,:,:,1));
            rat = overlap_ratio(occSamples{m}(1,:),occSamples{n}(1,:));
            if  rat > thr/2&&(( sim > thS && target_score{m}>0 && target_score{n}>0)&&(sum(ovBool(m,:))==0)&&(sum(ovBool(n,:))==0) || ovBool(m,n))
                if ovBool(m,n)==0
                    ovBool(m,n)=1;
                    ovBool(n,m)=1
                    img2 = imread(images{To-1});
                    fconv = mdnet_features_convX(net_conv, img2, result(m,To-1,:), opts);
                    occStart{m} = fconv(:,:,:,1);
                    fconv = mdnet_features_convX(net_conv, img2, result(n,To-1,:), opts);
                    occStart{n} = fconv(:,:,:,1);
                    SampStart{n} =result(n,To-1,:);
                    SampStart{m} =result(m,To-1,:);
                end
                ovConv{m}{end+1}{1}=occFrames{m}(:,:,:,1);
                ovConv{n}{end+1}{1}=occFrames{n}(:,:,:,1);
                ovSamp{m}{end+1}{1}=occSamples{m}(1,:);
                ovSamp{n}{end+1}{1}=occSamples{n}(1,:);

                if length(occSamples{n}(:,1))>1
                    for ind=2:length(occSamples{n})
                        if (overlap_ratio(occSamples{m}(1,:),occSamples{n}(ind,:))<thr) || cosine_sim(occFrames{m}(:,:,:,1),occFrames{n}(:,:,:,ind))<thS
                            ovConv{n}{end}{2}=occFrames{n}(:,:,:,ind);
                            ovSamp{n}{end}{2}=occSamples{n}(ind,:);
                            %occSamples{n}=occSamples(ind:end,:);
                            %occFrames{n}=occFrames(ind:end,:);
                            break;
                        end
                    end
                else
                    samples = gen_samples('gaussian', targetLoc(n,:), opts.nSamples, opts, trans_f(n), scale_f(n));
                    feat_conv = mdnet_features_convX(net_conv, img, samples, opts);
                    feat_fc = mdnet_features_fcX(net_fc{n}, feat_conv, opts);
                    feat_fc = squeeze(feat_fc)';
                    [scores,idx] = sort(feat_fc(:,2),'descend');
                    samples=samples(idx,:);
                    feat_conv=feat_conv(:,:,:,idx);
                    for ind=1:length(samples)
                        if (overlap_ratio(occSamples{m}(1,:),samples(ind,:))<thr) || cosine_sim(occFrames{m}(:,:,:,1), feat_conv(:,:,:,ind))<thS
                            ovConv{n}{end}{2}=feat_conv(:,:,:,ind);
                            ovSamp{n}{end}{2}=samples(ind,:);
                            occSamples{n}=samples(ind:end,:);
                            occFrames{n}=feat_conv(:,:,:,ind:end);
                            break;
                        end
                    end
                end
                if length(occSamples{m}(:,1))>1
                    for ind=2:length(occSamples{m})
                        if (overlap_ratio(occSamples{n}(1,:),occSamples{m}(ind,:))<thr) || cosine_sim(occFrames{n}(:,:,:,1),occFrames{m}(:,:,:,ind))<thS
                            ovConv{m}{end}{2}=occFrames{m}(:,:,:,ind);
                            ovSamp{m}{end}{2}=occSamples{m}(ind,:);
                            %occSamples{n}=occSamples(ind:end,:);
                            %occFrames{n}=occFrames(ind:end,:);
                            break;
                        end
                    end
                else
                    samples = gen_samples('gaussian', targetLoc(m,:), opts.nSamples, opts, trans_f(m), scale_f(m));
                    feat_conv = mdnet_features_convX(net_conv, img, samples, opts);
                    feat_fc = mdnet_features_fcX(net_fc{m}, feat_conv, opts);
                    feat_fc = squeeze(feat_fc)';
                    [scores,idx] = sort(feat_fc(:,2),'descend');
                    samples=samples(idx,:);
                    feat_conv=feat_conv(:,:,:,idx);
                    for ind=1:length(samples)
                        if (overlap_ratio(occSamples{n}(1,:),samples(ind,:))<thr) || cosine_sim(occFrames{n}(:,:,:,1), feat_conv(:,:,:,ind))<thS
                            ovConv{m}{end}{2}=feat_conv(:,:,:,ind);
                            ovSamp{m}{end}{2}=samples(ind,:);
                            occSamples{m}=samples(ind:end,:);
                            occFrames{m}=feat_conv(:,:,:,ind:end);
                            break;
                        end
                    end
                end
                %cosine_sim(occStart{m},ovConv{m}{end}{1})
                %cosine_sim(occStart{m},ovConv{m}{end}{2})
            elseif ovBool(m,n)
                ovBool(m,n)=0;
                ovBool(n,m)=0
                occEnd{m}=occFrames{m}(:,:,:,1);
                occEnd{n}=occFrames{n}(:,:,:,1);
                SampEnd{n} =result(n,To,:);
                SampEnd{m} =result(m,To,:);
                x=[m,n]
                [s,ind]=SimDP(occStart,ovConv,occEnd,ovSamp,SampStart,SampEnd,x);
                ind
                for i=1:length(ind)
                    frI = To-(length(ind)+1)+i;

                    for j=1:length(x)
                        result(x(j),frI,:)=ovSamp{x(j)}{i}{(j~=ind(i))+1};
                    end
                end
                tmp=result;
                for i=1:length(ind)
                    frI = To-(length(ind)+1)+i;
                    dist = min([3,frI-1,To-frI]);
                    for j=1:length(x)
                        
                        result(x(j),frI,:)= mean(tmp(x(j),frI-dist:frI+dist,:),2);
                    end
                    img2 = imread(images{frI});
                    if display
                        hc = get(gca, 'Children'); delete(hc(1:end-1));
                        set(hd,'cdata',img2); hold on;

                        for c = 1:M
                            rectangle('Position', result(c,frI,:), 'EdgeColor', colormap(mod(c,8)+1), 'Linewidth', 1);
                            text(result(c,frI,1),result(2,frI,2),num2str(c));
                        end
                        set(gca,'position',[0 0 1 1]);

                        text(10,10,[num2str(frI)],'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
                        hold off;
                        imwrite(frame2im(getframe(gcf)), [ pathSave num2str(frI) '.jpg']);
                        %imwrite(frame2im(getframe(gcf)), [ pathSave '5/' num2str(frI) '.jpg']);

                        drawnow;
                    end
                end
                ovConv{m}=cell(0);
                ovConv{n}=cell(0);
                ovSamp{m}=cell(0);
                ovSamp{n}=cell(0);
                SampStart{m}=cell(0);
                SampStart{n}=cell(0);
            end
        end
    end
    
    for m=1:M
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%% Occlusion Interpolation %%%%%%%%%%%%%%%%%%
        if (target_score{m}<0 && sum(ovBool(m,:)==0))
             if (occBool{m}==true)
%                 nSamp{m}=2;
                 occBool{m} = false;
                 occCount{m}=0;
             end
             occCount{m}=occCount{m}+1;   

        else
            if (occBool{m}==false)
%                nSamp{m}=5;
                occCount{m}=occCount{m}+1;
                occBool{m} = true;
                
                st=result(m,To-(occCount{m}),:);
                en=result(m,To,:);
                for i=1:(occCount{m}-1)
                    frI = To-(occCount{m})+i;
                    result(m,frI,:) = (i*en+(occCount{m}-i)*st)/occCount{m};
                    img = imread(images{frI});
                    if display
                        hc = get(gca, 'Children'); delete(hc(1:end-1));
                        set(hd,'cdata',img); hold on;

                        for c = 1:M
                            rectangle('Position', result(c,frI,:), 'EdgeColor', colormap(mod(c,8)+1), 'Linewidth', 1);
                            text(result(c,frI,1),result(c,frI,2),num2str(c));
                        end

                        set(gca,'position',[0 0 1 1]);

                        text(10,10,[num2str(frI)],'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
                        hold off;
                        imwrite(frame2im(getframe(gcf)), [ pathSave num2str(frI) '.jpg']);

                        drawnow;
                    end
                    
                end
                img = imread(images{To});
            end
        end
                    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                           % extend search space in case of failure
        if(target_score{m}<0)
                trans_f(m) = min(1.5, 1.1*trans_f(m));
        else
                trans_f(m) = opts.trans_f;
            end
       %% Prepare training data
        if(target_score{m}>0)&&sum(ovBool(m,:))==0
            pos_examples = gen_samples('gaussian', transpose(squeeze(result(m,To,:))), opts.nPos_update*2, opts, 0.1, 5);
            r = overlap_ratio(pos_examples,transpose(squeeze(result(m,To,:))));
            pos_examples = pos_examples(r>opts.posThr_update,:);
            pos_examples = pos_examples(randsample(end,min(opts.nPos_update,end)),:);

            neg_examples = gen_samples('uniform', transpose(squeeze(result(m,To,:))), opts.nNeg_update*2, opts, 2, 5);
            r = overlap_ratio(neg_examples,transpose(squeeze(result(m,To,:))));
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

        if((mod(To,opts.update_interval)==0 || (target_score{m}<0||sum(ovBool(m,:))~=0)) && To~=nFrames)
            
            if (target_score{m}<0||sum(ovBool(m,:))~=0) % short-term update
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
            rectangle('Position', result(m,To,:), 'EdgeColor', colormap(mod(m,8)+1), 'Linewidth',1);
            text(targetLoc(m,1),targetLoc(m,2),num2str(m));
        end
        
        %%rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
        set(gca,'position',[0 0 1 1]);
        
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        imwrite(frame2im(getframe(gcf)), [pathSave num2str(To) '.jpg']);
       
        drawnow;
    end
    
end