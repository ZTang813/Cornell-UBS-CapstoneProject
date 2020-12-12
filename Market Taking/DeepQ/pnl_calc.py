def PnLcalculator(actions, data):
    '''
    Arguments:
    actions: list of -1, 0, 1
    data: the data we perform actions on
    '''
    # act = {1:10,2:5,3:1,4:0,5:-1,6:-5,7:-10}
    # act = {1:5,2:1,3:0,4:-1,5:-5}
    time = 0
    position = 0
    unr_pnl = 0
    instant_pnl = 0
    pnl = []  ## This is what we return -- every episode's pnl (could be 0)
    history = []
    reward=0
    for i in range(1,len(actions) - 2):
        tomorrow_bid = data.iloc[time + 1, 1]
        tomorrow_ask = data.iloc[time + 1, 2]
        if position == 0:
            # either long or short will not have realized pnl
            if actions[i] == 1:
                unr_pnl = -tomorrow_ask
                position=1
            elif actions[i] == -1:
                unr_pnl = tomorrow_bid
                position=-1
            pnl.append(0)



        elif position == 1:
            # short will cause pnl, long will do nothing, flat will cause pnl
            if actions[i] == 0:
                unr_pnl += tomorrow_bid- data.iloc[time, 1]
                pnl.append(0)
            elif actions[i] == -1:
                position=0
                instant_pnl = tomorrow_bid + unr_pnl
                unr_pnl = 0
                reward+=instant_pnl
                pnl.append(instant_pnl)
            else:
                pnl.append(0)

        else:
            # long or flat will cause pnl, short will not
            if actions[i] == 0:
                unr_pnl -= tomorrow_ask-data.iloc[time, 2]
                pnl.append(0)
            elif actions[i] == 1:
                position=0
                instant_pnl = -tomorrow_ask + unr_pnl
                unr_pnl = 0
                reward+=instant_pnl
                pnl.append(instant_pnl)
            else:
                pnl.append(0)

        time += 1

    # final clearing
    if position == -1:
        reward += unr_pnl-data.iloc[time+1, 1]
        pnl.append(-data.iloc[time + 1, 1] + unr_pnl)
    elif position == 1:
        reward += float(data.iloc[time+1, 2]+unr_pnl)
        pnl.append(data.iloc[time + 1, 2] + unr_pnl)

    return reward, pnl
    # return pnl