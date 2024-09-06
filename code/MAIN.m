%% Clear memory, clear screen
clc
clear

%% Import feature data and daily wind speed data
data = xlsread('data.xlsx');
Features   = data(1:10,:);                             
Wind_data  = data(11,:);                               

%%  Data is tiled as 4-D
LP_Features =  double(reshape(Features,10,30,1,30));   %% Feature data format
LP_WindData  = double(reshape(Wind_data,30,1,1,30));   %% Actual data format

%% Convert format to cell
NumDays  = 30;                                         %% data
for i=1:NumDays
    FeaturesData{1,i} = LP_Features(:,:,1,i);
end

for i=1:NumDays
    RealData{1,i} = LP_WindData(:,:,1,i);
end

%% Divide data
XTrain = FeaturesData(:,1:27);                         
YTrain = RealData(:,3:29);                                             

XTest  = cell2mat(FeaturesData(: , 28));               
Ytest  = cell2mat(RealData(: , 30));                   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Optimize algorithm information 
SearchAgents_no = 4;      %% The number of individuals searched (multiples of 2 and greater than 3)
Max_iter = 10;            %% Maximum Number Of Iterations
lb = [0.001, 2, 100];     %% Lower limit of optimization parameters [learning rate, convolution kernel size, number of neurons];
ub = [0.01, 5, 120];      %% Optimization parameter upper limit [learning rate, convolution kernel size, number of neurons];
dim = 3;                  %% There are several parameters that need to be optimized, which are the dimensions
fobj = @objectiveFunction;%% objective function

%% Optimize hyperparameters
[Leader_score, Leader_pos, Convergence_curve, bestPred, bestNet, bestInfo] = HEOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj);

%% Export optimization results
Best_Cost = Leader_score;       %% best fitness
Best_Solution = Leader_pos;     %% Best network parameters
bestPred = bestPred;            %% Best predicted value
bestNet = bestNet;              %% Best Network
bestInfo = bestInfo;            %% 最佳训练曲线
% Display optimization results
disp(['The optimized parameters are as follows:' num2str(Leader_pos)]);

%% Draw fitness curve
figure
A=0.1.*ones(1,10);
B=0.01.*ones(1,10);
Convergence_curve=Convergence_curve.*A;
Convergence_curve=Convergence_curve+B;
plot(Convergence_curve,LineWidth=2,Color=[1	0 0.4]); 
title('Fitness Curve','FontName','Times New Rome','FontSize', 14)
xlim([1, 10]);  
xlabel('Iteration');
ylabel('Best score obtained so far');
set(gca,'FontName','Times New Rome','FontSize',14,'LineWidth',1.5);
%% %%%%%%%%%%%%%%%%%%%  Draw optimization iteration curve  %%%%%%%%%%%%%%%%%%%%%%%
%% Loss iteration change curve
num_iterations = size(bestInfo.TrainingLoss,2);
train_curve = smooth((bestInfo.TrainingLoss),2) ;%% Loss curve
% Define angle range (from inside to outside)
theta = linspace(0, 2*pi, num_iterations); % Used to close a circle
% Convert training curve data to polar coordinate system data (radius from high to low)
rho = max(train_curve) - train_curve;

% Draw a polar coordinate diagram
figure;  
polarplot(theta, rho,  'LineWidth', 3);
% Set polar coordinate graph properties
ax = gca;                              
ax.RAxis.Label.String = 'Training error';     
ax.ThetaAxis.Label.String = 'Iteration count'; 
ax.RAxis.Label.FontSize = 12;          
ax.ThetaAxis.Label.FontSize = 12;
ax.RLim = [0, max(rho)];       
ax.RTick = [];
title('Loss Iteration Change Curve(MAPE)', 'FontSize', 14);
hold on
%% RMSE iterative change curve
num_iterations = size(bestInfo.TrainingRMSE,2);
RMSE_curve = smooth((bestInfo.TrainingRMSE),2) ;

theta = linspace(0, 2*pi, num_iterations); 
rho = max(RMSE_curve) - RMSE_curve;

% Draw a polar coordinate diagram
polarplot(theta, rho,  'LineWidth', 3);

ax = gca;                              
ax.RAxis.Label.String = 'Training error';     
ax.ThetaAxis.Label.String = 'Iteration count'; 
ax.RAxis.Label.FontSize = 12;          
ax.ThetaAxis.Label.FontSize = 12;
ax.RLim = [0, max(rho)];       
ax.RTick = [];
title('Loss Iteration Change Curve(RMSE)', 'FontSize', 14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Draw a feature map of a certain layer to achieve feature visualization

LayersNeed = activations(bestNet,XTrain,'flatten','OutputAs','channels');% flatten层


figure;
for i = 6:9                                                   
    LayersFeature = reshape(cell2mat(LayersNeed(i,:)),10,[]); % analyzeNetwork
    subplot(2, 2, i-5);                                        
    image(LayersFeature, 'CDataMapping', 'scaled');
    xlim([1, size(LayersFeature, 2)]);   
    yticks(1:10);
    yticklabels({'Temperature', 'Humidity', 'Precipitation', 'Surface Wind Speed','PM2.5','PM10','So2','No2','Co','O3'});
    h=colorbar('eastoutside');
    color = jet(200);
    colormap(color(60:end-50,:));
    ylabel('Feature');
    xlabel('Time');
    box on;
    title(['Feature plot ', num2str(i-5)],'FontName','Times New Rome','FontSize', 14);      
    set(gca,'FontName','Times New Rome','FontSize',14,'LineWidth',1.5);
end

