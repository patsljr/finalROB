from t1func import Vector, substract, pdiv
import math
## DO NOT HAVE MOVEMENT VECTORS WITH 0 as a component, it will crash this 
def on_road(margin, pos_vec, vlis):
    if(pdiv(pos_vec, vlis[0].unit_vector()).nominal(margin) == True):
        return [True, vlis[0].unit_vector()]
    else:
        for i in range(len(vlis) - 2):
            pos_vec = substract(pos_vec, vlis[i])
            if(pdiv(pos_vec, vlis[i + 1].unit_vector()).nominal(margin) == True):
                if(pdiv(substract(pos_vec, vlis[i + 1]), vlis[i + 2].unit_vector()).nominal(margin)):
                    return [True, vlis[i + 2]]
                return [True, vlis[i + 1]]
        return False
## DO NOT HAVE MOVEMENT VECTORS WITH 0 as a component, it will crash this

##Note for today:
##Don't know what is going to happen if pos-vec is the intersection point of two vector
#This needs to be done to figure out what it's velocity vector should look like in order to turn
#Solution is to test next vector on succesful solution, then we know to use the next one instead
def get_vec_sol(margin, pos_vec, vlis):
    return on_road(margin, pos_vec, vlis)[1].angle()

def get_solution(margin, pos_vec, vlis):
    v = on_road(margin, pos_vec, vlis)
    if(v[0] ==  True):
        return v[1]
    else:
        return substract(v[1], pos_vec)

## return values for motors function
def rvm(goal_vector, current_vector, radi, factor,dv):
    diff = (goal_vector - current_vector)/(radi)(math.pi())
    if(-1 * math.pi()/12 < goal_vector - current_vector < math.pi()/12):
        return [dv, dv]
    elif(goal_vector - current_vector > 0):
        return [factor/diff**2, diff + factor/diff**2]
    else:
        return [diff + factor/diff**2, factor/diff**2]