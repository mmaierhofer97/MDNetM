%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%

clear;

conf = genConfig('mot','/MOT17-11-DPM/');
% conf = genConfig('vot2015','ball1');

switch(conf.dataset)
    case 'otb'
        net = fullfile('models','mdnet_vot-otb.mat');
    case 'vot2014'
        net = fullfile('models','mdnet_otb-vot14.mat');
    case 'vot2015'
        net = fullfile('models','mdnet_otb-vot15.mat');
    case 'mot'
        net = fullfile('models','mdnet_vot-otb_new.mat');
end
%% define the initialization for multiple targets. 2 as an example here
%conf.gt = [455,158,42,140]; %16-07
%conf.gt = [365,320,50,120];%;660,240,40,110;390,65,40,100;510,470,50,70]; %16-03
conf.gt = [868,256,84,120];%;948,153,154,624];  %17
 

conf
pathSave = [conf.imgDir(1:end-4) 'trackingM/'];
pathSave
result = mdnet_run(conf.imgList, conf.gt, net, 1, pathSave);
%[p,n] = SampGen(conf.imgList, conf.gt, net, 1, pathSave);
%%result = mdnet_run_ori(conf.imgList, conf.gt, net, 1);