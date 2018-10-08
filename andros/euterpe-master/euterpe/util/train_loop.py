def continue_train() :
    answer = True
    add_epoch = 0
    while True :
        answer = input("Continue training ? (yes / no) ")
        if answer not in ['yes', 'no'] :
            print("Wrong answer! Please type 'yes' OR 'no'")
            continue
        if answer == 'yes' :
            while True :
                try :
                    add_epoch = int(input("How many additional epoch ? (1 to N) "))
                    if add_epoch <= 0 :
                        print("Input must be and integer > 0")
                        continue
                except ValueError :
                    print("Input must be an integer")
                    continue
                else :
                    break
            break
        if answer == 'no' :
            break
    answer = True if answer == 'yes' else False
    return answer, add_epoch
