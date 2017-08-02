function [ result ] = mdnet_run(images, net, display, pathSave, det)
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

det = det(det(:,5) > 0 ,:);
iL = det(det(:,1)==1 & det(:,7) > 0 ,3:6);
initLoc=iL(1,:);
for i=2:length(iL(:,1))
    ov=1;
    for j=1:length(initLoc(:,1))
        if overlap_ratio(iL(i,:),initLoc(j,:))>0.4
            ov=0;
        end
    end
    if ov==1
        initLoc(end+1,:) = iL(i,:);
    end
end
%% Initialization
fprintf('Initialization...\n');

nFrames = length(images);
M = size(initLoc, 1);

img = imread(images{1});
if(size(img,3)==1), img = cat(3,img,img,img); end
justifi = ones(M, 1);
justifiF = ones(M, 1);
MoF = zeros(M,1);

targetLoc = initLoc;
result = zeros(M, nFrames, 4); result(:,1,:) = targetLoc;
%% Initialize displayots
%colormap = ['r','m','c','y','g','b','w','k'];
colormap = rand(M,3);
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    for m = 1:M
        rectangle('Position', targetLoc(m,:), 'EdgeColor', colormap(m,:), 'Linewidth', 1);
        text(targetLoc(m,1) + targetLoc(m,3)/2,targetLoc(m,2) + 12,num2str(m),'Color',colormap(m,:), 'FontSize', 18,'HorizontalAlignment', 'center' ); 
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
    total_pos_data{1,1,1,m,1} = feat_conv(:,:,:,pos_idx);
    total_neg_data{1,1,1,m,1} = feat_conv(:,:,:,neg_idx);

    success_frames{m} = [1];
    trans_f(m) = opts.trans_f;
    scale_f(m) = opts.scale_f;
end

save('16-11-set3')
%load('16-11-set3')

% Initialize displayots
colormap = ['r','m','c','y','g','b','w','k'];
colormap = rand(M,3);
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    for m = 1:M
        rectangle('Position', targetLoc(m,:), 'EdgeColor', colormap(m,:), 'Linewidth', 1);
        text(targetLoc(m,1) + targetLoc(m,3)/2,targetLoc(m,2) + 12,num2str(m),'Color',colormap(m,:), 'FontSize', 18,'HorizontalAlignment', 'center' ); 
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
ovBool=false;
ovCount=0;
ovConv = cell(M,1);
ovSamp = cell(M,1);
SampStart = cell(M,1);
SampEnd = cell(M,1);
ovF=cell(M);
%%%%%%%%%%%%%%

%% Main loop
for To = 2:min(nFrames)
    target_score;
    fprintf('Processing frame %d/%d... ', To, nFrames);
    
    img = imread(images{To});
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    detLoc = det(det(:,1)==To & det(:,7) > 0,3:6);
    spf = tic;
    %% Estimation
    for m = 1:M

        %check the detection result first
        if(justifiF(m) == 0)
            continue;
        end
        fprintf(num2str(m));
        prevLoc{m}=targetLoc(m,:);
        predLoc{m}=Trajectory_Pred(result(m,:,:),success_frames{m},To);
        r = overlap_ratio(detLoc,prevLoc{m});
        if target_score{m}>0
            o=full_over(detLoc,prevLoc{m});
        else
            o=zeros(size(detLoc(:,1)));
        end
        samples = detLoc(or((r > 0.7),(o==1)),:);
        if size(samples,1) > 0
            feat_conv = mdnet_features_convX(net_conv, img, samples, opts);
            feat_fc = mdnet_features_fcX(net_fc{m}, feat_conv, opts);
            feat_fc = squeeze(feat_fc)';
            [scores,idx] = sort(feat_fc(:,2),'descend');
            target_score{m} = scores(1);
        else
            target_score{m} = -1;
        end
        
        if target_score{m} > 0
            targetLoc(m,:) = samples(idx(1),:);
            result(m,To,:) = targetLoc(m,:);
            occFrames{m} = mdnet_features_convX(net_conv, img, targetLoc(m,:), opts);
            occSamples{m} = targetLoc(m,:);
            fprintf('Detection ');
        else
            fprintf('Tracking ');
            
            samples = gen_samples('gaussian',prevLoc{m}, opts.nSamples, opts, trans_f(m), scale_f(m));
            %samples = cat(1,samples,detLoc);
            feat_conv = mdnet_features_convX(net_conv, img, samples, opts);

            % evaluate the candidates
            feat_fc = mdnet_features_fcX(net_fc{m}, feat_conv, opts);
            feat_fc = squeeze(feat_fc)';
            [scores,idx] = sort(feat_fc(:,2),'descend');
            target_score{m} = mean(scores(1:5));
            targetLoc(m,:) = round(mean(samples(idx(1:5),:)));
            occFrames{m} = cat(4,mdnet_features_convX(net_conv, img, targetLoc(m,:), opts), feat_conv(:,:,:,idx));

            
            result(m,To,:) = targetLoc(m,:);
            occSamples{m} = [targetLoc(m,:);samples(idx,:)];
                    % extend search space in case of failure

            % bbox regression
            if(opts.bbreg && target_score{m}>0)
                X_ = permute(gather(feat_conv(:,:,:,idx(1:5))),[4,3,1,2]);
                X_ = X_(:,:);
                bbox_ = samples(idx(1:5),:);
                pred_boxes = predict_bbox_regressor(bbox_reg{m}.model, X_, bbox_);
                result(m,To,:) = round(mean(pred_boxes,1));
            end
        end
        
        %% Delete the branch
        
        % (1)delete the branch which is false for 120 times continuously
        if(target_score{m} >= 0)
            MoF(m,1) = 0;
        else
            if(MoF(m,1) > 120)||To==nFrames
                justifiF(m, 1) = 0;
                for frI = 1:MoF(m,1)
                    result(m,To+1-frI,:)=zeros(1,1,4);
                end
            else
                MoF(m,1) = MoF(m,1) + 1;
            end
        end
        
        % (2)Delete the branch which is out of boundary or suppress the neg score temporary
        temp = result(m,To,:);
        imi = max(temp(1), 0);
        jmi = max(temp(2), 0);
        ima = min(temp(1) + temp(3), size(img,2));
        jma = min(temp(2) + temp(4), size(img,1));
        state1 = imi <= 0;
        state2 = jmi <= 0;
        state3 = abs(ima - size(img,2)) <= 0;
        state4 = abs(jma - size(img,1)) <= 0;
        Out = 0;
        if(state1) Out = Out + 1; end
        if(state2) Out = Out + 1; end
        if(state3) Out = Out + 1; end
        if(state4) Out = Out + 1; end
        if(target_score{m} < 0 && Out == 0)% suppress the neg score temporary
        	justifi(m) = 0;
        	trans_f(m) = min(1.5, 1.1*trans_f(m));
        elseif(target_score{m} < 0 && Out > 2)
        	justifi(m) = 0;
        	justifiF(m) = 0;
        elseif (target_score{m} < 0 && Out <= 2)
        	if(justifi(m) ~= 0)
                justifi(m) = 0;
                justifiF(m) = 0;
            else % consider the case searching the occlusion situation
                result(m,To,:) = box_adjust(img, temp);
        	end
        else
        	justifi(m) = 1;
        	trans_f(m) = opts.trans_f;
        end
    end
        %%%%%%%%%%%%%%%%%%%%%% Overlap Logic %%%%%%%%%%%%%%%%%%%%%%%%
    for m=1:M-1
        if(justifiF(m) == 0)
            ovBool(m,:)=0;
            ovBool(:,m)=0;
            continue;
        end
        for n=m+1:M

            try ovBool(m,n)==0;
            catch ovBool(m,n)=0;
            end
            try ovBool(n,m)==0;
            catch ovBool(n,m)=0;
            end
            try length(ovF{m,n});
            catch ovF{m,n}=[];
            end
            try length(ovF{n,m});
            catch ovF{n,m}=[];
            end
            if(justifiF(n) == 0)
                ovBool(m,n)=0;
                ovBool(n,m)=0;
                continue;
            end
            thr=0.4;
            thS=0.6;
            sim = cosine_sim(occFrames{m}(:,:,:,1),occFrames{n}(:,:,:,1));
            rat = overlap_ratio(occSamples{m}(1,:),occSamples{n}(1,:));
            if  rat > thr/2&&(( sim > thS && target_score{m}>0 && target_score{n}>0)|| ovBool(m,n))
                if length(ovF{m,n})>50 || To==nFrames
                    ovBool(m,n)=0;
                    ovBool(n,m)=0;
                    if target_score{m}>target_score{n}
                        for frI = 1:length(ovF{m,n})
                            justifiF(n)=0;
                            result(n,ovF{m,n}(frI),:)=zeros(1,1,4);
                        end
                    else
                        for frI = 1:length(ovF{m,n})
                            justifiF(m)=0;
                            result(m,ovF{m,n}(frI),:)=zeros(1,1,4);
                        end
                    end
                    continue
                end
                if ovBool(m,n)==0
                    ovF{m,n}=[];
                    ovBool(m,n)=1;
                    ovBool(n,m)=1;
                    [m,n,1]
                    img2 = imread(images{To-1});
                    fconv = mdnet_features_convX(net_conv, img2, result(m,To-1,:), opts);
                    occStart{m}{To-1} = fconv(:,:,:,1);
                    fconv = mdnet_features_convX(net_conv, img2, result(n,To-1,:), opts);
                    occStart{n}{To-1} = fconv(:,:,:,1);
                    SampStart{n}{To-1} =result(n,To-1,:);
                    SampStart{m}{To-1} =result(m,To-1,:);
                end
                ovF{m,n}=[ovF{m,n},To];
                ovF{n,m}=ovF{m,n};
                ovConv{m}{To}{1}=occFrames{m}(:,:,:,1);
                ovConv{n}{To}{1}=occFrames{n}(:,:,:,1);
                ovSamp{m}{To}{1}=occSamples{m}(1,:);
                ovSamp{n}{To}{1}=occSamples{n}(1,:);

                if length(occSamples{n}(:,1))>1
                    for ind=2:size(occSamples{n},1)
                        
                        if (overlap_ratio(ovSamp{m}{To}{1},occSamples{n}(ind,:))<thr) || cosine_sim(ovConv{m}{To}{1}(:,:,:,1),occFrames{n}(:,:,:,ind))<thS
                            ovConv{n}{To}{2}=occFrames{n}(:,:,:,ind:end);
                            ovSamp{n}{To}{2}=occSamples{n}(ind:end,:);
                            occSamples{n}=occSamples{n}(ind:end,:);
                            occFrames{n}=occFrames{n}(:,:,:,ind:end);
                            break;
                        end
                    end
                else
                    samples = gen_samples('gaussian',prevLoc{n}, opts.nSamples, opts, trans_f(n), scale_f(n));
                    feat_conv = mdnet_features_convX(net_conv, img, samples, opts);
                    feat_fc = mdnet_features_fcX(net_fc{n}, feat_conv, opts);
                    feat_fc = squeeze(feat_fc)';
                    [scores,idx] = sort(feat_fc(:,2),'descend');
                    samples=samples(idx,:);
                    feat_conv=feat_conv(:,:,:,idx);
                    for ind=1:size(samples,1)
                        if (overlap_ratio(ovSamp{m}{To}{1},samples(ind,:))<thr) || cosine_sim(ovConv{m}{To}{1}(:,:,:,1), feat_conv(:,:,:,ind))<thS
                            ovConv{n}{To}{2}=feat_conv(:,:,:,ind:end);
                            ovSamp{n}{To}{2}=samples(ind:end,:);
                            occSamples{n}=samples(ind:end,:);
                            occFrames{n}=feat_conv(:,:,:,ind:end);
                            break;
                        end
                    end
                end
                if length(occSamples{m}(:,1))>1
                    for ind=2:size(occSamples{m},1)
                        if (overlap_ratio(ovSamp{n}{To}{1},occSamples{m}(ind,:))<thr) || cosine_sim(ovConv{n}{To}{1},occFrames{m}(:,:,:,ind))<thS
                            ovConv{m}{To}{2}=occFrames{m}(:,:,:,ind:end);
                            ovSamp{m}{To}{2}=occSamples{m}(ind:end,:);
                            occSamples{m}=occSamples{m}(ind:end,:);
                            occFrames{m}=occFrames{m}(:,:,:,ind:end);
                            break;
                        end
                    end
                else
                    samples = gen_samples('gaussian', prevLoc{m}, opts.nSamples, opts, trans_f(m), scale_f(m));
                    feat_conv = mdnet_features_convX(net_conv, img, samples, opts);
                    feat_fc = mdnet_features_fcX(net_fc{m}, feat_conv, opts);
                    feat_fc = squeeze(feat_fc)';
                    [scores,idx] = sort(feat_fc(:,2),'descend');
                    samples=samples(idx,:);
                    feat_conv=feat_conv(:,:,:,idx);
                    for ind=1:size(samples,1)
                        if (overlap_ratio(ovSamp{n}{To}{1},samples(ind,:))<thr) || cosine_sim(ovConv{n}{To}{1}, feat_conv(:,:,:,ind))<thS
                            ovConv{m}{To}{2}=feat_conv(:,:,:,ind:end);
                            ovSamp{m}{To}{2}=samples(ind:end,:);
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
                ovBool(n,m)=0;
                [m,n,0]
                occEnd{m}=occFrames{m}(:,:,:,1);
                occEnd{n}=occFrames{n}(:,:,:,1);
                SampEnd{n} =result(n,To,:);
                SampEnd{m} =result(m,To,:);
                x=[m,n];
                %if length(ovF{m,n})>1
                    [s,ind]=SimDP(occStart,ovConv,occEnd,ovSamp,SampStart,SampEnd,x,ovF{m,n});
                    y=1:M;
                    for i=2:length(ind)
                        frI = To-(length(ind)+1)+i;
                        for j=1:length(x)
                            if j~=ind(i)
                                yb=y(ovBool(x(j),:));
                                result(x(j),frI,:)=ovSamp{x(j)}{frI}{(j~=ind(i))+1}(1,:);
                                ovSamp{x(j)}{frI}{1}=ovSamp{x(j)}{frI}{(j~=ind(i))+1}(1,:);
                                ovConv{x(j)}{frI}{1}=ovConv{x(j)}{frI}{(j~=ind(i))+1}(:,:,:,1);
                                for br=1:length(yb)
                                    if any(ovF{x(j),yb(br)}==frI)
                                        
                                        for inc=1:size(ovSamp{yb(br)}{frI}{2},1)
                                            if (overlap_ratio(ovSamp{x(j)}{frI}{1},ovSamp{yb(br)}{frI}{2}(inc,:))<thr) || cosine_sim(ovConv{x(j)}{frI}{1}, ovConv{yb(br)}{frI}{2}(:,:,:,inc))<thS
                                                ovConv{yb(br)}{frI}{2}=ovConv{yb(br)}{frI}{2}(:,:,:,inc:end);
                                                ovSamp{yb(br)}{frI}{2}=ovSamp{yb(br)}{frI}{2}(inc:end,:);
                                                break;
                                            end
                                        end
                                        for inc=2:size(ovSamp{x(j)}{frI}{2},1)
                                            if (overlap_ratio(ovSamp{yb(br)}{frI}{1},ovSamp{x(j)}{frI}{2}(inc,:))<thr) || cosine_sim(ovConv{x(j)}{frI}{1}, ovConv{yb(br)}{frI}{2}(:,:,:,inc))<thS
                                                ovConv{x(j)}{frI}{2}=ovConv{yb(br)}{frI}{2}(:,:,:,inc:end);
                                                ovSamp{x(j)}{frI}{2}=ovSamp{x(j)}{frI}{2}(inc:end,:);
                                                break;
                                            end
                                        end
                                    end
                                end
                            end
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
                                if min(result(c,frI,3:4))<=0 
                                    continue;
                                end
                                rectangle('Position', result(c,frI,:), 'EdgeColor', colormap(c,:), 'Linewidth', 1);
                                text(result(c,frI,1) + targetLoc(m,3)/2,result(c,frI,2) + 12,num2str(c),'Color',colormap(c,:), 'FontSize', 18, 'HorizontalAlignment', 'center'); 
                            end

                            %%rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
                            set(gca,'position',[0 0 1 1]);

                            text(10,10,num2str(frI),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
                            hold off;
                            imwrite(frame2im(getframe(gcf)), [pathSave num2str(frI) '.jpg']);

                            drawnow;
                        end
                    end
                end
            %end
        end
    end

                
                
    for m=1:M
        if(justifiF(m) == 0)
            continue;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%% Occlusion Interpolation %%%%%%%%%%%%%%%%%%
        if (target_score{m}<0 && sum(ovBool(m,:)==0))
             if (occBool{m}==true)
%                 5=2;
                 occBool{m} = false;
                 occCount{m}=0;
             end
             occCount{m}=occCount{m}+1;   

        else
            if (occBool{m}==false)
%                5=5;
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
                            if min(result(c,frI,3:4))<=0 
                                continue;
                            end
                            rectangle('Position', result(c,frI,:), 'EdgeColor', colormap(c,:), 'Linewidth', 1);
                            text(result(c,frI,1) + targetLoc(m,3)/2,result(c,frI,2) + 12,num2str(c),'Color',colormap(c,:), 'FontSize', 18, 'HorizontalAlignment', 'center'); 
                        end

                        %%rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
                        set(gca,'position',[0 0 1 1]);

                        text(10,10,num2str(frI),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
                        hold off;
                        imwrite(frame2im(getframe(gcf)), [pathSave num2str(frI) '.jpg']);

                        drawnow;
                    end
                    
                end
                img = imread(images{To});
            end
        end
                    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                           % extend search space in case of failure

        %% Prepare training data
        if(target_score{m}>0&&sum(ovBool(m,:))==0)
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
            total_pos_data{1,1,1,m,To} = feat_conv(:,:,:,pos_idx);
            total_neg_data{1,1,1,m,To} = feat_conv(:,:,:,neg_idx);

            success_frames{m} = [success_frames{m}, To];
            if(numel(success_frames{m})>opts.nFrames_long)
                total_pos_data{1,1,1,m,success_frames{m}(end-opts.nFrames_long)} = single([]);
            end
            if(numel(success_frames{m})>opts.nFrames_short)
                total_neg_data{1,1,1,m,success_frames{m}(end-opts.nFrames_short)} = single([]);
            end
        else
            total_pos_data{1,1,1,m,To} = single([]);
            total_neg_data{1,1,1,m,To} = single([]);
        end

        %% Network update

        if((mod(To,opts.update_interval)==0 || (target_score{m}<0||sum(ovBool(m,:))~=0)) && To~=nFrames)
            
            if (target_score{m}<0||sum(ovBool(m,:))~=0) % short-term update
                temp_data = cell(1,1,1,min(opts.nFrames_short, size(success_frames{m},2)));
                temp_data(1,1,:,:) = total_pos_data(1,1,1,m,success_frames{m}(max(1,end-opts.nFrames_short+1):end));
                pos_data = cell2mat(temp_data);
            else % long-term update
                temp_data = cell(1,1,1,min(opts.nFrames_long, size(success_frames{m},2)));
                temp_data(1,1,:,:) = total_pos_data(1,1,1,m,success_frames{m}(max(1,end-opts.nFrames_long+1):end));
                pos_data = cell2mat(temp_data);
            end
            temp_data = cell(1,1,1,min(opts.nFrames_short, size(success_frames{m},2)));
            temp_data(1,1,:,:) = total_neg_data(1,1,1,m,success_frames{m}(max(1,end-opts.nFrames_short+1):end));
            neg_data = cell2mat(temp_data);

    %         fprintf('\n');
            net_fc{m} = mdnet_finetune_hnm(net_fc{m},pos_data,neg_data,opts,...
                'maxiter',opts.maxiter_update,'learningRate',opts.learningRate_update);
        end
        
    end
    
    %% get the unused detect bbox in current frame
    resultT = result(:,To,:);
    resultT = reshape(resultT, [], 4);
    decRes = setdiff(detLoc,resultT,'rows');

    %% Branch Increase
    if(mod(To,10) == 0 && ~isempty(decRes))
        if(M >= 5)
            % Filter the false positive situation
            decRestemp = zeros(size(decRes,1),1);
            len = size(decRes,1);
            for m = 1:M
                feat_conv_Res = mdnet_features_convX(net_conv, img, decRes, opts);
                feat_fc_c = mdnet_features_fcX(net_fc{m}, feat_conv_Res, opts);
                feat_fc_c = squeeze(feat_fc_c)';
                scores = feat_fc_c(:,2);
                for br = 1:size(decRes,1)
                    if(scores(br) > -0.75)
                        decRestemp(br,1) = decRestemp(br,1) + 1;
                    end
                end
            end
            decRes = decRes(decRestemp(:,1) >= max(len * 3 / 5, 1),:);
        end
        if(~isempty(decRes) && To ~= nFrames)
            % filter the repeated bounding box and add new branch
            for idx = 1: size(decRes,1)
                imi = max(decRes(idx,1), 0);
                jmi = max(decRes(idx,2), 0);
                ima = min(decRes(idx,1) + decRes(idx,3), size(img,2));
                jma = min(decRes(idx,2) + decRes(idx,4), size(img,1));
                stat1 = imi <= 1 || jmi <= 1 || ima <= 1 || jma <= 1;
                stat2 = abs(imi - size(img,2)) <= 1 || abs(ima - size(img,2)) <= 1 || abs(jmi - size(img,1)) <= 1 || abs(jma - size(img,1)) <= 1;
                if(stat1 || stat2)
                    continue;
                end
                situation = 0;
                for idM = 1:M
                    if 1==1%(justifi(idM, 1) == 1)
                        ra = max([overlap_ratio(decRes(idx,:), result(idM, To, :)),overlap_ratio(decRes(idx,:), predLoc{idM})]);
                        ba = (full_over(decRes(idx,:), result(idM, To, :))==1)||(full_over(decRes(idx,:), predLoc{idM})==1);
                        if(ra > 0.2||ba)
                            situation = 1;
                            break;
                        end
                    end
                end
                if(situation == 0)%initialize the new branch
                    M = M + 1;
                    justifi = [justifi; 1];
                    justifiF = [justifiF; 1];
                    MoF = [MoF; 0];
                    result(M, To, :) = decRes(idx,:);
                    % bbox train
                    pos_examples = gen_samples('uniform_aspect', decRes(idx,:), opts.bbreg_nSamples*10, opts, 0.3, 10);
                    r = overlap_ratio(pos_examples,decRes(idx,:));
                    pos_examples = pos_examples(r > opts.posThr_init,:);
                    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
                    feat_conv = mdnet_features_convX(net_conv, img, pos_examples, opts);
                    X = permute(gather(feat_conv),[4,3,1,2]);
                    X = X(:,:);
                    bbox = pos_examples;
                    bbox_gt = repmat(decRes(idx,:),size(pos_examples,1),1);
                    bbox_reg{M} = train_bbox_regressor(X, bbox, bbox_gt);
                    % draw positive/negative samples
                    pos_examples = gen_samples('gaussian', decRes(idx,:), opts.nPos_init*2, opts, 0.1, 5);
                    r = overlap_ratio(pos_examples,decRes(idx,:));
                    pos_examples = pos_examples(r>opts.posThr_init,:);
                    pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

                    neg_examples = [gen_samples('uniform', decRes(idx,:), opts.nNeg_init, opts, 1, 10);...
                                gen_samples('whole', decRes(idx,:), opts.nNeg_init, opts)];
                    r = overlap_ratio(neg_examples,decRes(idx,:));
                    neg_examples = neg_examples(r<opts.negThr_init,:);
                    neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

                    examples = [pos_examples; neg_examples];
                    pos_idx = 1:size(pos_examples,1);
                    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

                    % extract conv3 features
                    feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
                    pos_data = feat_conv(:,:,:,pos_idx);
                    neg_data = feat_conv(:,:,:,neg_idx);
                    % Learning CNN
                    fprintf('  training cnn...\n');
                    net_fc{M} = mdnet_finetune_hnm(net_fc_init,pos_data,neg_data,opts,...
                        'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);

                    neg_examples = gen_samples('uniform', decRes(idx,:), opts.nNeg_update*2, opts, 2, 5);
                    r = overlap_ratio(neg_examples,decRes(idx,:));
                    neg_examples = neg_examples(r<opts.negThr_init,:);
                    neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

                    examples = [pos_examples; neg_examples];
                    pos_idx = 1:size(pos_examples,1);
                    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

                    feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
                    total_pos_data{1,1,1,M,To} = feat_conv(:,:,:,pos_idx);
                    total_neg_data{1,1,1,M,To} = feat_conv(:,:,:,neg_idx);
                    
                    prevLoc{M}=result(M,To,:);
                    predLoc{M}=result(M,To,:);
                    success_frames{M} = [To];
                    trans_f(M) = opts.trans_f;
                    scale_f(M) = opts.scale_f;
                    targetLoc(M,:) = decRes(idx,:);
                    colormap(M,:) = rand(1,3);
                    occBool{M}=1;
                    target_score{M}=1;
                end
            end
        end
    end

    spf = toc(spf);
    fprintf('%f seconds\n',spf);
    %% Display
    if display
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',img); hold on;
        
        for m = 1:M
            if((justifi(m) == 0) || justifiF(m) == 0)%  && 
                continue;
            end
            rectangle('Position', result(m,To,:), 'EdgeColor', colormap(m,:), 'Linewidth', 1);
            %rectangle('Position', prevLoc{m}, 'EdgeColor', colormap(m,:), 'Linewidth', 1);
            text(result(m,To,1) + targetLoc(m,3)/2,result(m,To,2) + 12,num2str(m),'Color',colormap(m,:), 'FontSize', 18, 'HorizontalAlignment', 'center'); 
        end
        
        %%rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
        set(gca,'position',[0 0 1 1]);
        
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        imwrite(frame2im(getframe(gcf)), [pathSave num2str(To) '.jpg']);
       
        drawnow;
    end
end
for To = 2:min(nFrames)
    img = imread(images{To});
    if display
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',img); hold on;
        
        for m = 1:M
            if min(result(m,To,3:4))<=0 
                continue;
            end
            rectangle('Position', result(m,To,:), 'EdgeColor', colormap(m,:), 'Linewidth', 1);
            %rectangle('Position', prevLoc{m}, 'EdgeColor', colormap(m,:), 'Linewidth', 1);
            text(result(m,To,1) + targetLoc(m,3)/2,result(m,To,2) + 12,num2str(m),'Color',colormap(m,:), 'FontSize', 18, 'HorizontalAlignment', 'center'); 
        end
        
        %%rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
        set(gca,'position',[0 0 1 1]);
        
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        imwrite(frame2im(getframe(gcf)), [pathSave num2str(To) '.jpg']);
       
        drawnow;
    end
end