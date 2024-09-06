function [ Best_score, BEST_pos, Convergence_curve, bestPred, bestNet, bestInfo] = HEOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
% SearchAgents_no, Max_iter, lb, ub, dim, fobj
if (max(size(ub)) == 1)
    ub = ub .* ones(1, dim);
    lb = lb .* ones(1, dim);
end

% 初始化随机解集
for i = 1:dim
    X(:, i) = lb(i) + rand(SearchAgents_no, 1) .* (ub(i) - lb(i));   % 初始种群
end

Best_pos = zeros(1, dim);
Best_score = inf;
Objective_values = zeros(1, size(X, 1));

Convergence_curve = [];
N1 = floor(SearchAgents_no * 0.5);
Elite_pool = [];

% 计算第一个解集的适应度并找到最佳解
for i = 1:size(X, 1)
    [Objective_values(1, i),Value{1,i},Net{1,i},Info{1,i}] = fobj(X(i, :));
    if i == 1
        Best_pos = X(i, :);
        % 除了学习率其它位置均为整数
        for t = 1:size(Best_pos,2)
            if t ==1
                BEST_pos(t) =  Best_pos(t);
            else
                BEST_pos(t) =  round(Best_pos(t));
            end  
        end

        Best_score = Objective_values(1, i);
        bestPred = Value{1,i};
        bestNet = Net{1,i};
        bestInfo= Info{1,i};
    elseif Objective_values(1, i) < Best_score
        Best_pos = X(i, :);
        % 除了学习率其它位置均为整数
        for t = 1:size(Best_pos,2)
            if t ==1
                BEST_pos(t) =  Best_pos(t);
            else
                BEST_pos(t) =  round(Best_pos(t));
            end  
        end
        Best_score = Objective_values(1, i);
        bestPred = Value{1,i};
        bestNet = Net{1,i};
        bestInfo= Info{1,i};
    end
    
    All_objective_values(1, i) = Objective_values(1, i);
end

[~, idx1] = sort(Objective_values);
second_best = X(idx1(2), :);
third_best = X(idx1(3), :);
sum1 = 0;
for i = 1:N1
    sum1 = sum1 + X(idx1(i), :);
end
half_best_mean = sum1 / N1;
Elite_pool(1, :) = Best_pos;
Elite_pool(2, :) = second_best;
Elite_pool(3, :) = third_best;
Elite_pool(4, :) = half_best_mean;

Convergence_curve(1) = Best_score;

for i = 1:SearchAgents_no
    index(i) = i;
end

Na = SearchAgents_no / 2;
Nb = SearchAgents_no / 2;

% 主循环
l = 2; % 从第二次迭代开始，因为第一次迭代用于计算适应度
while l <= Max_iter
    RB = randn(SearchAgents_no, dim);          %布朗随机数向量
    T = exp(-l / Max_iter);
    k = 1;
    DDF = 0.4 * (1 + (3 / 5) * (exp(l / Max_iter) - 1)^k / (exp(1) - 1)^k);
    M = DDF * T;
    
    %% 计算整个种群的质心位置
    for j = 1:dim
        sum1 = 0;
        for i = 1:SearchAgents_no
            sum1 = sum1 + X(i, j);
        end
        X_centroid(j) = sum1 / SearchAgents_no;
    end
    
    %% 随机选择个体构建 pop1 和 pop2
    index1 = randperm(SearchAgents_no, Na);
    index2 = setdiff(index, index1);
    
    for i = 1:Na
        r1 = rand;
        k1 = randperm(4, 1);
        for j = 1:size(X, 2) % 在第 j 维
            X(index1(i), j) = Elite_pool(k1, j) + RB(index1(i), j) * (r1 * (Best_pos(j) - X(index1(i), j)) + (1 - r1) * (X_centroid(j) - X(index1(i), j)));
        end
    end
    
    if Na < SearchAgents_no
        Na = Na + 1;
        Nb = Nb - 1;
    end
    
    if Nb >= 1
        for i = 1:Nb
            r2 = 2 * rand - 1;
            for j = 1:size(X, 2) % 在第 j 维
                X(index2(i), j) = M * Best_pos(j) + RB(index2(i), j) * (r2 * (Best_pos(j) - X(index2(i), j)) + (1 - r2) * (X_centroid(j) - X(index2(i), j)));
            end
        end
    end
    
    % 检查解是否超出搜索空间，并将其拉回
    for i = 1:size(X, 1)
        for j = 1:dim
            if X(i, j) > ub(j)
                X(i, j) = ub(j);
            end
            if X(i, j) < lb(j)
                X(i, j) = lb(j);
            end
        end
        
        % 计算目标值
        [Objective_values(1, i),Value{1,i},Net{1,i},Info{1,i}] = fobj(X(i, :));
        % 如果有更好的解，则更新目标值
        if Objective_values(1, i) < Best_score
            Best_pos = X(i, :);
            % 除了学习率其它位置均为整数
            for t = 1:size(Best_pos,2)
                if t ==1
                    BEST_pos(t) =  Best_pos(t);
                else
                    BEST_pos(t) =  round(Best_pos(t));
                end  
            end

            Best_score = Objective_values(1, i);
            bestPred = Value{1,i};
            bestNet = Net{1,i};
            bestInfo= Info{1,i};
        end
    end
    
    %% 更新精英池
    [~, idx1] = sort(Objective_values);
    second_best = X(idx1(2), :);
    third_best = X(idx1(3), :);
    sum1 = 0;
    for i = 1:N1
        sum1 = sum1 + X(idx1(i), :);
    end
    half_best_mean = sum1 / N1;
    Elite_pool(1, :) = Best_pos;
    Elite_pool(2, :) = second_best;
    Elite_pool(3, :) = third_best;
    Elite_pool(4, :) = half_best_mean;

    Convergence_curve(l) = Best_score;
    l = l + 1;
end
