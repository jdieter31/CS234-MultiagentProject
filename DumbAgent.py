import AgentInterface as inter
import time

def run(team_name, number, first_player):
    interface = inter.AgentInterface(team_name, number)
    if first_player:
        interface.set_home(3, 2)
    else:
        interface.set_home(4, 2)
    obs = interface.observe_from_server()
    while ("start", 0) not in obs:
        obs = interface.observe_from_server()

    while ("stop", 0) not in obs:
        new_cycle = False
        for o in obs:
            if o[0] == "cycle":
                new_cycle = True
                obs = []
                break
        if new_cycle:
            interface.send_action("move", 1)
        obs = interface.observe_from_server()