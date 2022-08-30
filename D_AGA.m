% city_coordinate=tsp(N);%城市位置
[N,coordinates]=Read('kroA100.tsp');
t1=clock;     %计算运行时间
city_coordinate = coordinates;

s=100;%样本数
c=0.2;%精英保留比例
pc=0.9;%交叉概率
pm=0.2;%变异概率
times=10000;%最大迭代次数
time=0;%实际迭代次数
pop=zeros(s,N+1);%初始种群+适应度
pop_fit_aver=[];%总适应度
min_dis=[];%最短距离
pop_min=[];%最短距离的基因
city_distance=CityDistance(city_coordinate,N);%城市间距离
% a1=MVODM(city_distance);%MVODM距离矩阵改造
for i=1:s   %初始化
    sb=ceil(rand*N);
%     [~,path]=dijkstra(a1,sb);
            [~,path]=dijkstra(city_distance,sb);%未改造之前的距离矩阵
    pop(i,1:N)=path;
end

[individual_fit,sum,min1,min_index]=GroupFit(city_distance,N,pop,s);%适应度
sumP=sum;
pop_fit_aver=[pop_fit_aver;sum];    %总适应度
min_dis=[min_dis;min1];             %最短距离
pop(:,N+1)=individual_fit;          %适应度为距离倒数
pop_min=[pop_min;pop(min_index,:)]; %最短距离基因，即最短距离index
for i=1:times
    time=time+1;
    pop5=pop;   %父代
    E_new_new=[];   %子代
    for j=1:s/2
        a=rand(1);
        bj=min(individual_fit(j,1),individual_fit(j+s/2,1));
        min11=min(individual_fit);
        if bj<=sum/s
            pc1=pc*(min(individual_fit(j,1),individual_fit(j+s/2,1))-min11)/(sum/s-min11);
        elseif bj>sum/s
            pc1=pc;
        end
        if a>pc1          %交叉
        else
            crosspoint=rand(1,2);
            crosspoint=floor(crosspoint*N)+1;
            [pop(j,:),pop(j+s/2,:)]=CrossVariation(pop(j,:),pop(j+s/2,:),crosspoint,N);
        end
    end
    for j=1:s
        b=rand(1);
        bj=individual_fit(j,1);
        min11=min(individual_fit);
        if bj<=sum/s
            pm1=pm*(individual_fit(j,1)-min11)/(sum/s-min11);
        elseif bj>sum/s
            pm1=pm;
        end
        if b>pm1          %变异
        else
            pop(j,:)=Mutation(pop(j,:),N);
        end
        E_new_new=[E_new_new;pop(j,:)];
    end
    [individual_fit,sum,min1,min_index]=GroupFit(city_distance,N,E_new_new,s);
    sumS=sum;
    pop_fit_aver=[pop_fit_aver;sum];
    min_dis=[min_dis;min1];
    E_new_new(:,N+1)=individual_fit;
    pop_min=[pop_min;E_new_new(min_index,:)];
    if(abs(sumS-sumP)<eps)%退出条件
        break;
    end
    pop=ChooseParents(pop5,E_new_new,N,s,c);
end
[a,min_index]=min(min_dis);

DrawPath(city_coordinate,pop_min,min_index,N);

function [n_citys,city_position] = Read(filename)
fid = fopen(filename,'rt');
location=[];
A = [1 2];
tline = fgetl(fid);%fgetl读取文件第一行
while ischar(tline)
    if(strcmp(tline,'NODE_COORD_SECTION'))%比较文本是否为数据开头
        while ~isempty(A)
            A=fscanf(fid,'%f',[3,1]);%读取文本文件中的数据，fileID,formatSpec,sizeA
            if isempty(A)
                break;
            end
            location=[location;A(2:3)'];
        end
    end
    tline = fgetl(fid); %fgetl继续读取文件下一行
    if strcmp(tline,'EOF')%比较文本是否为数据结束
        break;
    end
end
[m,n]=size(location);
n_citys = m;
city_position=location;
fclose(fid);
end

function [individual_fit,num,min_distance,a]=GroupFit(city_distance,N,pop,s)%种群适应度
individual_distance=zeros(s,1);
for j=1:s
    sum_distance=0;
    for i=1:N-1
        sum_distance=sum_distance+city_distance(pop(j,i),pop(j,i+1));   %不闭合
    end
    sum_distance=sum_distance+city_distance(pop(j,N),pop(j,1));    %闭合
    individual_distance(j,1)=sum_distance;
end
[min_distance,a]=min(individual_distance);
individual_fit=1./individual_distance;
num=0;
for i=1:s
    num=num+individual_fit(i,1);
end
end
function [pop_ok]=ChooseParents(pop,pop1,N,s,c)%选择父代
    pop=sortrows(pop,N+1,'descend');  %父代根据适应度从大到小排序
    pop1=sortrows(pop1,N+1,'descend');  %子代根据适应度从大到小排序
    pop2=[pop1(1:s*(1-c),:);pop(1:s*c,:)];
    randIndex=randperm(size(pop2,1));    %打乱选择后的排序
    pop2=pop2(randIndex,:);
    pop_ok=pop2;
end
function [a,b]=SwapRepeat(tbl,pop1,pop2,c1,c2,N)%基因去重
    i=100/N;
    for k=1:(c1-1)
        if tbl(pop1(k),3)>i
            kk=find(pop1(c1:c2)==pop1(k))+c1-1; %找到C1-C2间重复的位置
            kkk=pop1(k);        %找到C1前重复的位置
            pop1(k)=pop2(kk);
            pop2(kk)=kkk;
        end
    end
    for k=c2+1:N
        if tbl(pop1(k),3)>i
            kk=find(pop1(c1:c2)==pop1(k))+c1-1;
            kkk=pop1(k);
            pop1(k)=pop2(kk);
            pop2(kk)=kkk;
        end
    end
    a=pop1;
    b=pop2;
end
function [a,b]=CrossVariation(pop1,pop2,crosspoint,N)%交叉
    A=pop1;
    if(crosspoint(:,1)<crosspoint(:,2))
        pop1(crosspoint(:,1):crosspoint(:,2))=pop2(crosspoint(:,1):crosspoint(:,2));
        pop2(crosspoint(:,1):crosspoint(:,2))=A(1,crosspoint(:,1):crosspoint(:,2));
        while 1
            tbl = tabulate(pop1(1:N));  %创建向量频率表，第一列为unique后的数据，第二列为频数、第三列为出现频率
            if (tbl(:,3)<=(100/N))  %没有重复
                break;
            end
            [pop1,pop2]=SwapRepeat(tbl,pop1,pop2,crosspoint(:,1),crosspoint(:,2),N);
        end
    else
        pop1(crosspoint(:,2):crosspoint(:,1))=pop2(crosspoint(:,2):crosspoint(:,1));
        pop2(crosspoint(:,2):crosspoint(:,1))=A(1,crosspoint(:,2):crosspoint(:,1));
        while 1
            tbl = tabulate(pop1(1:N));
            if (tbl(:,3)<=(100/N))
                break;
            end
            [pop1,pop2]=SwapRepeat(tbl,pop1,pop2,crosspoint(:,2),crosspoint(:,1),N);
        end
    end
    a=pop1;b=pop2;
end
function [a]=SwapGene(sub,c1,c2)%交换，倒序
    kk=ceil((c2-c1)/2);
    kkk=(c2-c1)+2;
    for k=1:kk
        kkkk=sub(k);
        sub(k)=sub(kkk-k);
        sub(kkk-k)=kkkk;
    end
    a=sub;
end
function [a]=Mutation(pop0,N)%变异
    crosspoint=rand(1,2);
    crosspoint=floor(crosspoint*N)+1;
    if(crosspoint(:,1)<crosspoint(:,2))
        sub=pop0(crosspoint(:,1):crosspoint(:,2));
        sub=SwapGene(sub,crosspoint(:,1),crosspoint(:,2));
        pop0(crosspoint(:,1):crosspoint(:,2))=sub;
    else
        sub=pop0(crosspoint(:,2):crosspoint(:,1));
        sub=SwapGene(sub,crosspoint(:,2),crosspoint(:,1));
        pop0(crosspoint(:,2):crosspoint(:,1))=sub;
    end
    a=pop0;
end
function DrawPath(city_coordinate,E_new_new,min_index,N)%画路径图
    k=E_new_new(min_index,1:N);
    plot(city_coordinate(:,1),city_coordinate(:,2),'r*','MarkerSize',10,'linewidth',1);
    hold on;
    for i=1:N-1
        plot([city_coordinate(k(i),1),city_coordinate(k(i+1),1)],[city_coordinate(k(i),2),city_coordinate(k(i+1),2)],'b','LineWidth',1);
        set(gcf,'color','white');
    end
    xlabel('X轴/mm')
    ylabel('Y轴/mm')
    set(gca,'linewidth',0.5,'fontsize',15)
    box off 
end
function [distance,path]=dijkstra(a,sb)
% 输入：a—距离矩阵
% sb—起点的标号
% 输出：distance—短路的距离, path—短路的路径
n=size(a,1);         %找出节点个数n
visited=zeros(n,1);      %构建初步的查找矩阵visited，0代表未查找，1代表查找过
visited(sb)=1;
distance=0;
path=zeros(1,n);
path(1,1)=sb;
for i=1:n-1
    flag=visited==1;
    q=a(sb,:);
    q(flag)=inf;
    [value,location]=min(q);
    distance=distance+value;
    path(1,i+1)=location;
    visited(location)=1;
    sb=location;
end
end%贪心算法初始化

function [city_distance]=CityDistance(city_coordinate,N)%城市距离矩阵
    city_distance=zeros(N,N);
    vx=14.5;vy=42; % mm/ms
    tx=0;ty=0; % ms
    coor_x_tmp1 = city_coordinate(:,1) * ones(1,N);
    coor_x_tmp2 = coor_x_tmp1';
    coor_y_tmp1 = city_coordinate(:,2) * ones(1,N);
    coor_y_tmp2 = coor_y_tmp1';
    % dist_matrix = sqrt((coor_x_tmp1-coor_x_tmp2).^2 + ...
    % (coor_y_tmp1-coor_y_tmp2).^2);
    lx = abs((coor_x_tmp1-coor_x_tmp2))/vx + tx;
    ly = abs((coor_y_tmp1-coor_y_tmp2))/vy + ty;
    lx(lx==tx)=0;
    ly(ly==ty)=0;
    city_distance = max(lx,ly);
end