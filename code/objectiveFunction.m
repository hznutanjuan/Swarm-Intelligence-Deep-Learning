function [R,tsmvalue,net, info] = objectiveFunction(x)
%% 导入
data = xlsread('data.xlsx');
Features   = data(1:10,:);                             %% 特征输入 
Wind_data  = data(11,:);                               %% 实际值输出

%%  数据平铺为4-D
LP_Features =  double(reshape(Features,10,30,1,30));   %% 特征数据格式
LP_WindData  = double(reshape(Wind_data,30,1,1,30));   %% 实际数据格式

%% 格式转换为cell
NumDays  = 30;                                         %% 数
for i=1:NumDays
    FeaturesData{1,i} = LP_Features(:,:,1,i);
end

for i=1:NumDays
    RealData{1,i} = LP_WindData(:,:,1,i);
end

%% 划分数据
XTrain = FeaturesData(:,1:27);                         %% 训练集输入
YTrain = RealData(:,3:29);                             %% 训练集输出                

XTest  = cell2mat(FeaturesData(: , 28));               %% 测试集输入
Ytest  = cell2mat(RealData(: , 30));                   %% 测试集输出

%% 将优化目标参数传进来的值 转换为需要的超参数
learning_rate = x(1);            %% 学习率
KerlSize = round(x(2));          %% 卷积核大小
NumNeurons = round(x(3));        %% 神经元个数

%% 网络搭建
lgraph = layerGraph();

% 添加层分支
% 将网络分支添加到层次图中。每个分支均为一个线性层组。
tempLayers = sequenceInputLayer([10 30 1],"Name","sequence");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer(KerlSize,32,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    maxPooling2dLayer([3 3],"Name","maxpool","Padding","same")
    flattenLayer("Name","flatten_1")
    fullyConnectedLayer(25,"Name","fc_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = flattenLayer("Name","flatten");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = gruLayer(NumNeurons,"Name","gru1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    FlipLayer("flip3")
    gruLayer(NumNeurons,"Name","gru2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(1,3,"Name","concat")
    selfAttentionLayer(1,50,"Name","selfattention")
    fullyConnectedLayer(30,"Name","fc")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

% 连接层分支
% 连接网络的所有分支以创建网络图。
lgraph = connectLayers(lgraph,"sequence","conv");
lgraph = connectLayers(lgraph,"sequence","flatten");
lgraph = connectLayers(lgraph,"flatten","gru1");
lgraph = connectLayers(lgraph,"flatten","flip3");
lgraph = connectLayers(lgraph,"gru1","concat/in1");
lgraph = connectLayers(lgraph,"gru2","concat/in2");
lgraph = connectLayers(lgraph,"fc_1","concat/in3");

%% 设置训练参数
options = trainingOptions('adam', ...   % adam 梯度下降算法
    'MaxEpochs',300, ...                % 最大训练次数 300
    'GradientThreshold',1,...           % 渐变的正阈值 1
    'ExecutionEnvironment','cpu',...    % 网络的执行环境 cpu
    'InitialLearnRate',learning_rate,...% 初始学习率 0.01
    'LearnRateSchedule','none',...      % 训练期间降低整体学习率的方法 不降低
    'Shuffle','every-epoch',...         % 每次训练打乱数据集
    'SequenceLength',30,...             % 序列长度 20
    'MiniBatchSize',10,...              % 训练批次大小 每次训练样本个数20
    'Verbose',true);                    % 有关训练进度的信息不打印到命令窗口中

%% 训练网络
[net,info] = trainNetwork(XTrain,YTrain, lgraph, options);

%% 测试与评估
YPredicted = net.predict(XTest);                       
tsmvalue = YPredicted;

%% 计算误差
% 过程
error2 = YPredicted-Ytest;            % 测试值和真实值的误差  
[~,len]=size(Ytest);                  % len获取测试样本个数，数值等于testNum，用于求各指标平均值
SSE1=sum(error2.^2);                  % 误差平方和
MAE1=sum(abs(error2))/len;            % 平均绝对误差
MSE1=error2*error2'/len;              % 均方误差
RMSE1=MSE1^(1/2);                     % 均方根误差
MAPE1=mean(abs(error2./mean(Ytest))); % 平均百分比误差
r=corrcoef(Ytest,YPredicted);         % corrcoef计算相关系数矩阵，包括自相关和互相关系数
R1=r(1,2); 
R=MAPE1;
display(['本批次MAPE:', num2str(R)]);
end

