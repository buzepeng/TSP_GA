import numpy as np
from matplotlib import pyplot as plt

class TSP_GA():
    def __init__(self, filepath = "./TSPlib/kroA100.tsp", s = 100, c = 0.2, pc = 0.9, pm = 0.2, times = 10000):
        self.s, self.c, self.pc, self.pm, self.times = s, c, pc, pm, times
        self.Read(filepath)
        self.CityDistance()
        self.pop = np.zeros((s, self.N+1), dtype = np.float32)
        for i in range(s):
            sb = np.random.randint(0, self.N)
            self.pop[i,:-1] = self.dijkstra(self.city_distance, sb)

    def Read(self, filename):
        pos = []
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                sub_line = line.split(" ", -1)
                if(sub_line[0].isdigit()):
                    pos.append([int(sub_line[1]), int(sub_line[2])])
        # return len(pos), np.array(pos)
        self.N = len(pos)
        self.city_coordinate = np.array(pos)

    def CityDistance(self):
        self.city_distance = np.zeros((self.N, self.N), dtype = np.float32)
        vx, vy = 14.5, 42
        tx, ty = 0, 0
        coor_x_tmp1 = self.city_coordinate[:, 0].reshape(-1,1) * np.ones((1, self.N))
        coor_x_tmp2 = coor_x_tmp1.transpose()
        coor_y_tmp1 = self.city_coordinate[:, 1].reshape(-1,1) * np.ones((1, self.N))
        coor_y_tmp2 = coor_y_tmp1.transpose()

        lx = abs(coor_x_tmp1 - coor_x_tmp2)/vx + tx
        ly = abs(coor_y_tmp1 - coor_y_tmp2)/vy + ty
        lx[lx == tx]=0
        ly[ly == ty]=0
        self.city_distance = np.where(lx>ly, lx, ly)

    def dijkstra(self, a, sb):
        n = a.shape[0]
        a = a.copy()
        visited = np.zeros(n, dtype = np.bool)
        visited[sb] = 1
        path = np.zeros(n, dtype = np.int)
        path[0] = sb
        for i in range(n-1):
            q = a[sb,:]
            q[visited] = np.inf
            location = np.argmin(q)
            path[i+1] = location
            visited[location] = True
            sb = location
        return path

    def GroupFit(self, city_distance, N, pop, s):
        pop = pop.copy().astype(int)
        individual_distance = np.sum(city_distance[pop[:, np.arange(N-1)], pop[:, np.arange(1, N)]], axis = 1) + city_distance[pop[:,N-1], pop[:,0]]
        a = np.argmin(individual_distance)
        min_distance = individual_distance[a]
        individual_fit=1./individual_distance
        num = np.sum(individual_fit)
        return individual_fit, num, min_distance, a

    def ChooseParents(self, pop, pop1, N, s, c):
        pop, pop1 = pop.copy(), pop1.copy()
        pop = pop[np.argsort(pop[:, -1])[::-1], :]
        pop1 = pop1[np.argsort(pop1[:,-1])[::-1], :]
        pop2 = np.r_[pop1[:int(s*(1-c)),:],pop[:int(s*c),:]]
        randInd = np.random.permutation(pop2.shape[0])
        return  pop2[randInd,:]

    # def SwapRepeat(self, tbl, pop1, pop2, c1, c2, N):
    #     pop1, pop2 = pop1.copy(), pop2.copy()
    #     i = 100.0/N
    #     for k in np.append(np.arange(c1), np.arange(c2, N)):
    #         if tbl[int(pop1[k]),2]>i:
    #             kk = np.where(pop1[c1:c2]==pop1[k])[0][0]+c1
    #             kkk = pop1[k]
    #             pop1[k] = pop2[kk]
    #             pop2[kk] = kkk
    #     return pop1, pop2

    # def CrossVariation(pop1, pop2, crosspoint, N):
    #     tmp, pop1, pop2 = pop1.copy(), pop1.copy(), pop2.copy()
    #     min_cross, max_cross = np.min(crosspoint), np.max(crosspoint)
    #     pop1[min_cross:max_cross] = pop2[min_cross:max_cross]
    #     pop2[min_cross:max_cross] = tmp[min_cross:max_cross]
    #     tbl = np.zeros((N,3))
    #     tbl[:,0] = np.arange(N)
    #     count = 0
    #     while True:
    #         uniq, freq = np.unique(pop1[:N],return_counts=True)
    #         uniq = uniq.astype(np.int32)
    #         tbl[uniq,1] = freq
    #         tbl[uniq,2] = freq
    #         if((tbl[:,2]<=100.0/N).all()):
    #             break
    #         pop1, pop2 = SwapRepeat(tbl, pop1, pop2, min_cross, max_cross, N)
    #         count += 1
    #     return pop1, pop2

    # def check_perm(self, pop, N):
    #     pop = pop.copy()
    #     return (pop.sort() == np.arange(N)).all()

    def CrossVariation(self, pop1, pop2, crosspoint, N):
        # pop1, pop2 = list(pop1[:-1]), list(pop2[:-1])
        crossoverPoint1, crossoverPoint2 = np.min(crosspoint), np.max(crosspoint)
        child1, child2 = -1*np.ones(N+1, dtype = np.float32), -1*np.ones(N+1, dtype = np.float32) 
        child1[crossoverPoint1:crossoverPoint2] = pop1[crossoverPoint1:crossoverPoint2]
        child2[crossoverPoint1:crossoverPoint2] = pop2[crossoverPoint1:crossoverPoint2]
        # print(self.check_perm(pop1[:-1], self.N), self.check_perm(pop2[:-1], self.N))
        for t in pop2[crossoverPoint1:crossoverPoint2]:
            if t not in pop1[crossoverPoint1:crossoverPoint2]:
                # ind = pop2.index(t)
                # ind = lookup_table2[t]
                ind = np.where(pop2[:-1] == t)[0]
                # print(t, ind)
                while crossoverPoint1 <= ind and ind < crossoverPoint2:
                    temp = pop1[ind]
                    # ind = pop2.index(temp)
                    # ind = lookup_table2[temp]
                    ind = np.where(pop2[:-1] == temp)[0]
                child1[ind] = t

        ind = np.where(child1 == -1)[0]
        child1[ind] = pop2[ind]

        
        for t in pop1[crossoverPoint1:crossoverPoint2]:
            if t not in pop2[crossoverPoint1:crossoverPoint2]:
                # ind = np.where(pop1[:N] == t)
                # ind = lookup_table1[t]
                ind = np.where(pop1[:-1] == t)[0]
                while crossoverPoint1 <= ind and ind < crossoverPoint2:
                    temp = pop2[ind]
                    # ind = np.where(pop1[:N] == temp)
                    # ind = lookup_table1[temp]
                    ind = np.where(pop1[:-1] == temp)[0]
                child2[ind] = t
    
        ind = np.where(child2 == -1)
        child2[ind] = pop1[ind]
        child1[-1], child2[-1] = pop1[-1], pop2[-1]
        return np.array(child1), np.array(child2)

    def Mutation(self, pop0, N):
        pop0 = pop0.copy()
        crosspoint = np.random.randint(0, N, 2)
        min_cross, max_cross = np.min(crosspoint), np.max(crosspoint)
        pop0[min_cross:max_cross] = np.flip(pop0[min_cross:max_cross])
        return pop0

    def DrawPath(self, city_coordinate, E_new_new, N):
        k = E_new_new[:N].astype(int)
        plt.scatter(city_coordinate[:,0], city_coordinate[:,1], marker = '*', c ='red')
        for i in range(N-1):
            plt.plot([city_coordinate[k[i], 0], city_coordinate[k[i+1], 0]], [city_coordinate[k[i],1], city_coordinate[k[i+1], 1]])
        plt.xlabel('X/mm')
        plt.ylabel('Y/mm')
        plt.box(False)
        plt.show()

    def FindShortestPath(self):
        c, s, N, times, pc, pm = self.c, self.s, self.N, self.times, self.pc, self.pm

        individual_fit,sumP,min_dis,min_index=self.GroupFit(self.city_distance, self.N, self.pop, s)
        sum = sumP
        self.pop[:,-1] = individual_fit
        self.pop_min = self.pop[min_index]

        for time in range(times):
            pop5 = self.pop.copy()
            # E_new_new = np.array([])
            for j in range(s//2):
                a = np.random.randn()
                bj = min(individual_fit[j], individual_fit[j+s//2])
                min11 = np.min(individual_fit)
                if bj<=sum/s:
                    pc1 = pc*(bj-min11)/(sum/s-min11)
                else:
                    pc1 = pc
                if a<=pc1:
                    crosspoint = np.random.randint(0, N, 2)
                    self.pop[j,:], self.pop[j+s//2,:] = self.CrossVariation(self.pop[j,:], self.pop[j+s//2,:], crosspoint, N)
            for j in range(s):
                b = np.random.randn()
                bj = individual_fit[j]
                min11 = np.min(individual_fit)
                if bj<=sum/s:
                    pm1 = pm*(individual_fit[j]-min11)/(sum/s-min11)
                else:
                    pm1 = pm
                if b<=pm1:
                    self.pop[j,:] = self.Mutation(self.pop[j,:],N)
            #     E_new_new = np.append(E_new_new, pop[j,:])
            # E_new_new = E_new_new.reshape(s, -1)
            # individual_fit,sum,min1,min_index = GroupFit(city_distance,N,E_new_new,s)
            individual_fit,sum,min1,min_index = self.GroupFit(self.city_distance,N,self.pop,s)
            sumS = sum
            self.pop[:, -1] = individual_fit
            # E_new_new[:,N] = individual_fit
            # pop[:,-1] = individual_fit
            if min1<min_dis:
                # pop_min = E_new_new[min_index,:]
                self.pop_min = self.pop[min_index,:]
                self.min_dis = min1
            if(np.abs(sumS-sumP)<1e-8):
                break
            self.pop = self.ChooseParents(pop5, self.pop, N, s, c)
            print(self.pop[:,-1])
        self.DrawPath(self.city_coordinate, self.pop_min, N)


if __name__ == "__main__":
    tsp_ga = TSP_GA()
    tsp_ga.FindShortestPath()