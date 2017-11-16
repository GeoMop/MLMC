import time as t
import SimulationSetting as Sim
import Result
import MLMC as MLMC


class Main:
    '''
    Class launchs MLMC
    '''

    def main(*args):

        start = t.time()
        Y, n, v, N, cas = ([] for i in range(5))

        pocet_urovni = 4
        # for 1 ,2 failed
        pocet_vykonani = 3

        result = Result.Result()
        result.set_levels_number(pocet_urovni)
        result.set_execution_number(pocet_vykonani)


        #sim = SimW.SimulationWater(10,10)
        #sim = Sim.SimulationShooting(np.array([0, 0]), np.array([10, 0]), np.array([-100, 200, -300, 400])j, 10, np.array([0, -1]))

        # param type -> type of simulation
        sim = Sim.SimulationSetting(1)

        result = Result.Result()
        result.set_levels_number(pocet_urovni)
        result.set_execution_number(pocet_vykonani)


        for i in range (pocet_vykonani):
            start_MC = t.time()
            # number of levels, n_fine, n_coarse, simulation
            m = MLMC.MLMC(pocet_urovni, 1000, 10, sim)

            # type, time or variance
            m.monte_carlo(1, 0.01)
            end_MC = t.time()

            result.add_average(m.get_Y())

            result.add_arrays(m.get_arrays())

            result.append_result(m.formatting_result())
            #n.append(P[0])
            #print("mala n", P[0])
            #v.append(P[1])
            #print("rozptyly V", P[1])
            #N.append(P[2])
            #print("velka N", P[2])
            result.add_time(end_MC - start_MC)
            #cas.append(end_MC - start_MC)
            #print("Cas jedne MLMC", end_MC - start_MC)

        result.result()




        '''
        sum = 0
        stredni_hodnota = 50
        for i in range(1):
            sum = sum + pow((stredni_hodnota -Y[i]),2)
        chyba = np.sqrt(sum/10)
        end = t.time()
        
        print("Y = ", m.average(Y))
        print("Chyba = ", chyba)
        
        print("cas", end - start)
        '''
     


    if __name__ == "__main__":
        main()
