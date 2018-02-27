import socket

TCP_IP = '127.0.0.1'
TCP_PORT = 5050
BUFFER_SIZE = 1024


# Class for interacting with grid soccer simulator
class AgentInterface:
    # Attributes: (Set After Initialization/receiving updates from server
    # team_name - string - team_name
    # uni_number - integer - uniform number
    # left_team - boolean - true if on left team, false if on right
    # rows - integer - number of rows on field
    # cols - integer - number of columns on field
    # goal_width - integer - width of goal
    # pass_dist - integer - max pass distance configured by server
    # visible_dist - integer - distance visible as set by server
    # min_players - integer - minimum number of players on a team
    # max_players - integer - maximum number of players on a team
    # home_row - integer - row of home location (set_home must be called first for this to be valid)
    # home_col - integer - col of home location (set_home must be called first for this to be valid)
    # row - integer - current row of agent
    # col - integer - current column of agent

    # Initializes Agent With Server
    # team_name - One Word String Containing Team Name
    # uni_number - Integer Containing Uniform Number
    def __init__(self, team_name, uni_number):
        self.team_name = team_name
        self.uni_number = uni_number

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((TCP_IP, TCP_PORT))
        self.sock.send("(init " + team_name + " " + str(uni_number) + ")")

        init_response = self.sock.recv(BUFFER_SIZE)
        print(init_response)
        if (init_response == "(init l ok)"):
            self.left_team = True
        elif (init_response == "(init r ok)"):
            self.left_team = False
        else:
            raise IOError("Unexpected Response From Server")

        settings_response = self.sock.recv(BUFFER_SIZE)
        print(settings_response)

    def parse_settings(self, settings_msg):
        # TODO
        return

    # Returns a dictionary of data retrieved from the server
    # where the keys are strings representing the observation type and the values are data relating
    # to each key. Valid key value pairs are listed below:
    # KEY: "ball" VALUE: [ball_row, ball_col] - The location of the ball was seen
    # KEY: "player" VALUE: [left_team (boolean), uni_number, row, col] - The location of a player was seen
    # KEY: "start" VALUE: 0 - The game was started
    # KEY: "stop" VALUE: 0 - The game was stopped
    # KEY: "turbo" VALUE: [turbo_on (boolean)] The turbo setting was turned on/off
    # KEY: "cycle" VALUE: [new_cycle_length] The cycle length (in milliseconds) was adjusted
    def observe_from_server(self):
        # TODO
        return

    # Sets the agent's home position to the given row and column
    def set_home(self, row, col):
        #TODO
        return

    # Sends an action to the server valid actions are:
    # action: move, action_data: int representing direction (0 - hold, 1 - east, 2 - south, 3 - west, 4 - north,
    # 5 - north-east, 6 - south-east, 7 - south-west, 8, north-west)
    # action: pass, action_data: <target_uni_number>
    def send_action(self, action, action_data=None):
        #TODO
        return