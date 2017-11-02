import numpy as np
import random

# Seria interessante receber os parametros antes da execucao
ind_size = 100   # Size of every individual
input_amount = 5 # Amount of inputs
func_amount = 2  # Amount of functions STILL NEED TO DEFINE THESE
pop_size = 5     # Population Size
# Talvez as operações devessem ser já formatos de árvores (uniao de varios nos)..

def create_new_individual():
    new_ind = {}
    new_ind['genotype'] = []
    possible_nodes = list(range(0, input_amount)) # Possible nodes to be add as inputs / output
    operators = list(range(0, func_amount))
    last = input_amount -1;
    for i in range(0, ind_size):
        #tmp_value = random.choice(possible_nodes)
        new_ind['genotype'].append(random.choice(possible_nodes))
        #if tmp_value < input_amount:
        #    possible_values.pop(tmp_value)
        new_ind['genotype'].append(random.choice(possible_nodes))
        new_ind['genotype'].append(random.choice(operators))
        last+=1
        possible_nodes.append(last)
    new_ind['output'] = random.choice(possible_nodes[input_amount:])
    #new_ind['fitness'] = evaluate(new_ind)
    return new_ind

# Separa os nós ativos presentes.
def active_nodes(ind):
    out = ind['output']
    gen = ind['genotype']
    to_eval=[]
    to_eval.append(out)
    evaluated = list(range(0, input_amount))
    used_nodes = {}
    inicial_position = input_amount
    while(len(to_eval)>0):
        if(to_eval[0] in evaluated):
            to_eval.pop(0)
        else:
            tmp = []
            for i in range(0,3):
                value = gen[(to_eval[0]-inicial_position)*3 + i]
                tmp.append(value)
                if((value not in evaluated) and (i!=2)):
                    to_eval.append(value)
            used_nodes[to_eval[0]] = tmp
            evaluated.append(to_eval[0])
    return used_nodes

'''
# input: individuo, matriz de distancias.
# output: 'fitness'
def evaluate(ind)


#input: individuo para ser mutado, chance de mutação.
#output: novo individuo mutado
def mutation(ind)


def create_population():
    pop = []
    for i in range(pop_size):
        pop.append(create_new_individual())
    return pop

#input: population, numero max de geracoes.
#output: best individual (solution)
def reproduction():
    # ordena os individuos
    # Cria 4 mutacoes do mais apto.
    # re-calcula o fitness
    # passa pra proxima geração.


'''


ind = create_new_individual()
act = active_nodes(ind)
print(ind)
print(act)
