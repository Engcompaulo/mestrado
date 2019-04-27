
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
    
TRAIN_SIZE = 0.8 

VAR_LO = 'Variance Baixa'
VAR_ME = 'Variance Média'
VAR_HI = 'Variance Alta'

SKE_LO = 'Skewness Baixa'
SKE_ME = 'Skewness Média'
SKE_HI = 'Skewness Alta'

CUR_LO = 'Custosis Baixa'
CUR_ME = 'Custosis Média'
CUR_HI = 'Custosis Alta'

ENT_LO = 'Entropy Baixa'
ENT_ME = 'Entropy Média'
ENT_HI = 'Entropy Alta'

def generate_uniform_mf():

    my_data = np.genfromtxt('./docs/dados_autent_bancaria.txt', delimiter=',')
    variance_data = my_data[:,0]
    skewness_data = my_data[:,1]
    curtosis_data = my_data[:,2]
    entropy_data = my_data[:,3]

    offset = 1

    variance = np.arange(int(min(variance_data)-offset), int(max(variance_data)+offset), 0.01)
    skewness = np.arange(int(min(skewness_data)-offset), int(max(skewness_data)+offset), 0.01)
    curtosis = np.arange(int(min(curtosis_data)-offset), int(max(curtosis_data)+offset), 0.01)
    entropy = np.arange(int(min(entropy_data)-offset), int(max(entropy_data)+offset), 0.01)
    classif = np.arange(0, 1, 0.01)

    vmax = max(variance)
    vmin = min(variance)

    std = np.std(variance_data)
    qtd = (vmax-vmin)/std

    var_lo = fuzz.trimf(variance, [vmin, vmin + qtd*0.15*std, vmin + qtd*0.30*std])
    var_me = fuzz.trimf(variance, [vmin + qtd*0.15*std, vmin + qtd*0.50*std, vmin + qtd*0.85*std])
    var_hi = fuzz.trimf(variance, [vmin + qtd*0.70*std, vmin + qtd*0.85*std, vmax])

    vmax = max(skewness)
    vmin = min(skewness)

    std = np.std(skewness_data)
    qtd = (vmax-vmin)/std

    ske_lo = fuzz.trimf(skewness, [vmin, vmin + qtd*0.15*std, vmin + qtd*0.30*std])
    ske_me = fuzz.trimf(skewness, [vmin + qtd*0.15*std, vmin + qtd*0.50*std, vmin + qtd*0.85*std])
    ske_hi = fuzz.trimf(skewness, [vmin + qtd*0.70*std, vmin + qtd*0.85*std, vmax])

    vmax = max(curtosis)
    vmin = min(curtosis)
    
    std = np.std(curtosis_data)
    qtd = (vmax-vmin)/std

    cur_lo = fuzz.trimf(curtosis, [vmin, vmin + qtd*0.15*std, vmin + qtd*0.30*std])
    cur_me = fuzz.trimf(curtosis, [vmin + qtd*0.15*std, vmin + qtd*0.50*std, vmin + qtd*0.85*std])
    cur_hi = fuzz.trimf(curtosis, [vmin + qtd*0.70*std, vmin + qtd*0.85*std, vmax])

    vmax = max(entropy)
    vmin = min(entropy)
    
    std = np.std(entropy_data)
    qtd = (vmax-vmin)/std

    ent_lo = fuzz.trimf(entropy, [vmin, vmin + qtd*0.15*std, vmin + qtd*0.30*std])
    ent_me = fuzz.trimf(entropy, [vmin + qtd*0.15*std, vmin + qtd*0.50*std, vmin + qtd*0.85*std])
    ent_hi = fuzz.trimf(entropy, [vmin + qtd*0.70*std, vmin + qtd*0.85*std, vmax])

    return (variance, skewness, curtosis, entropy, classif,
            var_lo, var_me, var_hi, 
            ske_lo, ske_me, ske_hi, 
            cur_lo, cur_me, cur_hi, 
            ent_lo, ent_me, ent_hi)

def activated_rule(var_value, skew_value, cur_value, ent_value,  
                   variance, skewness, curtosis, entropy,  
                   var_lo, var_me, var_hi, 
                   ske_lo, ske_me, ske_hi, 
                   cur_lo, cur_me, cur_hi, 
                   ent_lo, ent_me, ent_hi):      
    activation_var = []
    activation_var.append((VAR_LO, fuzz.interp_membership(variance, var_lo, var_value)))
    activation_var.append((VAR_ME, fuzz.interp_membership(variance, var_me, var_value)))
    activation_var.append((VAR_HI, fuzz.interp_membership(variance, var_hi, var_value)))
    activation_var.sort(key=lambda var: var[1], reverse=True)
    var_max = activation_var[0]
    
    activation_ske = []
    activation_ske.append((SKE_LO, fuzz.interp_membership(skewness, ske_lo, skew_value)))
    activation_ske.append((SKE_ME, fuzz.interp_membership(skewness, ske_me, skew_value)))
    activation_ske.append((SKE_HI, fuzz.interp_membership(skewness, ske_hi, skew_value)))
    activation_ske.sort(key=lambda ske: ske[1], reverse=True)
    max_ske = activation_ske[0]
    
    activation_cur = []
    activation_cur.append((CUR_LO, fuzz.interp_membership(curtosis, cur_lo, cur_value)))
    activation_cur.append((CUR_ME, fuzz.interp_membership(curtosis, cur_me, cur_value)))
    activation_cur.append((CUR_HI, fuzz.interp_membership(curtosis, cur_hi, cur_value)))
    activation_cur.sort(key=lambda cur: cur[1], reverse=True)
    max_cur = activation_cur[0]
    
    activation_ent = []
    activation_ent.append((ENT_LO, fuzz.interp_membership(entropy, ent_lo, ent_value)))
    activation_ent.append((ENT_ME, fuzz.interp_membership(entropy, ent_me, ent_value)))
    activation_ent.append((ENT_HI, fuzz.interp_membership(entropy, ent_hi, ent_value)))
    activation_ent.sort(key=lambda ent: ent[1], reverse=True)
    max_ent = activation_ent[0]

    rule_activation = min(var_max[1],max_ske[1],max_cur[1],max_ent[1])    
    
    return (var_max[0],max_ske[0],max_cur[0],max_ent[0]),rule_activation

def generate_rules(data, 
              variance, skewness, curtosis, entropy,  
              var_lo, var_me, var_hi, 
              ske_lo, ske_me, ske_hi, 
              cur_lo, cur_me, cur_hi, 
              ent_lo, ent_me, ent_hi):
    rules = {} 
    for d in data:        
        rule, rule_activation, = activated_rule(d[0], d[1], d[2], d[3], variance, skewness, curtosis, entropy, var_lo, var_me, var_hi, ske_lo, ske_me, ske_hi, cur_lo, cur_me, cur_hi, ent_lo, ent_me, ent_hi)
        if rule not in rules:
            rules[rule] = rule_activation, d[4]
        else:
            if rule_activation > rules[rule][1]:
                rules[rule] = rule_activation, d[4]
    return rules

def activation(var_value, skew_value, cur_values, ent_value, 
              variance, skewness, curtosis, entropy,  
              var_lo, var_me, var_hi, 
              ske_lo, ske_me, ske_hi, 
              cur_lo, cur_me, cur_hi, 
              ent_lo, ent_me, ent_hi):
    activations = {}
    
    activations[VAR_LO] = fuzz.interp_membership(variance, var_lo, var_value)
    activations[VAR_ME] = fuzz.interp_membership(variance, var_me, var_value)
    activations[VAR_HI] = fuzz.interp_membership(variance, var_hi, var_value)
    
    activations[SKE_LO] = fuzz.interp_membership(skewness, ske_lo, skew_value)
    activations[SKE_ME] = fuzz.interp_membership(skewness, ske_me, skew_value)
    activations[SKE_HI] = fuzz.interp_membership(skewness, ske_hi, skew_value)

    activations[CUR_LO] = fuzz.interp_membership(curtosis, cur_lo, cur_values)
    activations[CUR_ME] = fuzz.interp_membership(curtosis, cur_me, cur_values)
    activations[CUR_HI] = fuzz.interp_membership(curtosis, cur_hi, cur_values)

    activations[ENT_LO] = fuzz.interp_membership(entropy, ent_lo, ent_value)
    activations[ENT_ME] = fuzz.interp_membership(entropy, ent_me, ent_value)
    activations[ENT_HI] = fuzz.interp_membership(entropy, ent_hi, ent_value)
                
    return activations

def fuzzify(rules, activations, classif):
    output_ativations = []    
    for r,v in rules.items():
        var, ske, cur, ent = r       
        active_rule = max(activations[var], activations[ske], activations[cur], activations[ent])
        if(v[1] == 0):
            output_ativations.append(np.fmin(active_rule, fuzz.trimf(classif, [0, 0.3, 0.6])))
        else:
            output_ativations.append(np.fmin(active_rule, fuzz.trimf(classif, [0.5, 0.75, 1])))

    return output_ativations

def defuzzify(output_ativations, classif):
    aggregated = 0
    for out in output_ativations:
        aggregated = np.fmax(aggregated, out)

    value = fuzz.defuzz(classif, aggregated, 'mom')
    return value

def generate_train_test_data():    
    my_data = np.genfromtxt('./docs/dados_autent_bancaria.txt', delimiter=',')
    np.random.shuffle(my_data)

    true_data = [d for d in my_data if d[4] == 0]
    false_data = [d for d in my_data if d[4] == 1]

    train_data = true_data[:int(len(true_data)*TRAIN_SIZE)] + false_data[:int(len(false_data)*TRAIN_SIZE)]
    test_data = true_data[int(len(true_data)*TRAIN_SIZE):] + false_data[int(len(false_data)*TRAIN_SIZE):]
    
    return train_data,test_data    

def main():

    (variance, skewness, curtosis, entropy, classif,
    var_lo, var_me, var_hi, 
    ske_lo, ske_me, ske_hi, 
    cur_lo, cur_me, cur_hi, 
    ent_lo, ent_me, ent_hi) = generate_uniform_mf()

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(12, 10))

    ax0.plot(variance, var_lo, 'b', linewidth=1.5, label='Baixa')
    ax0.plot(variance, var_me, 'g', linewidth=1.5, label='Média')
    ax0.plot(variance, var_hi, 'y', linewidth=1.5, label='Alta')
    ax0.set_title('Variance')
    ax0.legend(loc=5)

    ax1.plot(skewness, ske_lo, 'b', linewidth=1.5, label='Baixa')
    ax1.plot(skewness, ske_me, 'g', linewidth=1.5, label='Média Baixa')
    ax1.plot(skewness, ske_hi, 'y', linewidth=1.5, label='Alta')
    ax1.set_title('Skewness')
    ax1.legend(loc=5)

    ax2.plot(curtosis, cur_lo, 'b', linewidth=1.5, label='Baixa')
    ax2.plot(curtosis, cur_me, 'g', linewidth=1.5, label='Média')
    ax2.plot(curtosis, cur_hi, 'y', linewidth=1.5, label='Alta')
    ax2.set_title('Curtosis')
    ax2.legend(loc=5)

    ax3.plot(entropy, ent_lo, 'b', linewidth=1.5, label='Baixa')
    ax3.plot(entropy, ent_me, 'g', linewidth=1.5, label='Média')
    ax3.plot(entropy, ent_hi, 'y', linewidth=1.5, label='Alta')
    ax3.set_title('Entropy')
    ax3.legend(loc=5)

    ax4.plot(classif, fuzz.trimf(classif, [0, 0.3, 0.6]), 'b', linewidth=1.5, label='Falsa')
    ax4.plot(classif, fuzz.trimf(classif, [0.5, 0.75, 1]), 'y', linewidth=1.5, label='Verdadeira')
    ax4.set_title('Classification')
    ax4.legend(loc=5)

    for ax in (ax0, ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    fig.savefig('fuzzy-var-hist.png')

    train_data, test_data = generate_train_test_data()

    rules = generate_rules(train_data, 
              variance, skewness, curtosis, entropy,  
              var_lo, var_me, var_hi, 
              ske_lo, ske_me, ske_hi, 
              cur_lo, cur_me, cur_hi, 
              ent_lo, ent_me, ent_hi)

    acertos = 0
    erros = 0

    for data in test_data:
        activations = activation(data[0], data[1], data[2], data[3], 
                                variance, skewness, curtosis, entropy,  
                                var_lo, var_me, var_hi, 
                                ske_lo, ske_me, ske_hi, 
                                cur_lo, cur_me, cur_hi, 
                                ent_lo, ent_me, ent_hi)
        output_ativations = fuzzify(rules, activations, classif)
        class_value = 1 if defuzzify(output_ativations, classif) >= 0.6 else 0 
        if(class_value==data[4]):
            acertos += 1
        else:
            erros += 1    
        print('O modelo classificou em {} o que era {} - ACERTOU: {}'.format(class_value,data[4],class_value==data[4]))

    print('Acertos: {}, Erros: {}'.format(acertos, erros))

if __name__ == "__main__":
    main()