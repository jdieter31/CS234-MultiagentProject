import socket
import re
import select

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

    # Initializes Agent With Server
    # team_name - One Word String Containing Team Name
    # uni_number - Integer Containing Uniform Number
    def __init__(self, team_name, uni_number):
        self.team_name = team_name
        self.uni_number = uni_number

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((TCP_IP, TCP_PORT))
        self.sock.send("(init " + team_name + " " + str(uni_number) + ")")

        msg = self.receive_messages()

        # Make sure we received settings message too
        if len(msg) < 2:
            second_msg = self.receive_messages()
            if len(second_msg) == 0:
                raise IOError("Unexpected Response From Server")
            msg.append(second_msg[0])

        init_response = msg[0]
        if (init_response == "(init l ok)"):
            self.left_team = True
        elif (init_response == "(init r ok)"):
            self.left_team = False
        else:
            raise IOError("Unexpected Response From Server")

        settings_response = msg[1]
        self.parse_settings(settings_response)

    def receive_messages(self):
        # Make sure socket is readable to not block execution trying to read
        r, w, e = select.select([self.sock], [], [], 1)
        if len(r) == 0:
            return []

        data = self.sock.recv(BUFFER_SIZE)
        # Split by null terminators
        array = data.split('\x00')

        #filter out edge-case empty strings
        array = filter(None, array)

        return array

    def parse_settings(self, settings_msg):
        prog = re.compile("\(settings \(rows (?P<rows>\w+)\) \(cols (?P<cols>\w+)\) \(goal-width (?P<goal_width>\w+)\) \(pass-dist (?P<pass_dist>\w+)\) \(visible-dist (?P<visible_dist>\w+)\) \(min-players (?P<min_players>\w+)\) \(max-players (?P<max_players>\w+)\)\)")
        result = prog.match(settings_msg)
        if result is None:
            raise IOError("Unexpected Response From Server")
        self.rows = int(result.group("rows"))
        self.cols = int(result.group("cols"))
        self.goal_width = int(result.group("goal_width"))
        self.pass_dist = int(result.group("pass_dist"))
        self.visible_dist = int(result.group("visible_dist"))
        self.min_players = int(result.group("min_players"))
        self.max_players = int(result.group("max_players"))
        return

    # Returns a list of data retrieved from the server
    # where each element is a tuple with the first element representing the observation type and the second the data relating
    # to each observation. Valid observations types are listed below
    # ("score", [left_score, right_score])
    # ("loc", [row, col]) - The agents current location
    # ("ball", [ball_row, ball_col]) - The location of the ball was seen
    # ("player", [left_team (boolean), uni_number, row, col]) - The location of a player was seen
    # ("start", 0) - The game was started
    # ("stop", 0) - The game was stopped
    # ("cycle", cycle_number) - New timestep in the game
    # ("turbo", turbo_on (boolean)) - The turbo setting was turned on/off
    # ("cycle_length", new_cycle_length) - The cycle length (in milliseconds) was adjusted
    def observe_from_server(self):
        msgs = self.receive_messages()
        obs = []
        for msg in msgs:
            if msg == "(start)":
                obs.append(("start", 0))
                continue
            if msg == "(stop)":
                obs.append(("stop", 0))
                continue
            cycle_prog = re.compile("\(cycle (?P<cycle_length>\w+)\)")
            cycle_match = cycle_prog.match(msg)
            if cycle_match is not None:
                obs.append(("cycle_length", int(cycle_match.group("cycle_length"))))
                continue

            turbo_prog = re.compile("\(turbo (?P<turbo_on>\w+)\)")
            turbo_match = turbo_prog.match(msg)
            if turbo_match is not None:
                if turbo_match.group("turbo_on") == "on":
                    obs.append(("turbo", True))
                else:
                    obs.append(("turbo", False))
                continue

            see_prog = re.compile("\(see (?P<cycle_number>\w+) (?P<see_data>.+)\)\Z")
            see_match = see_prog.match(msg)
            if see_match is not None:
                obs.append(("cycle", int(see_match.group("cycle_number"))))
                see_data = see_match.group("see_data")
                agent_prog = re.compile("\(self (?P<row>\w+) (?P<col>\w+)\)")
                agent_match = agent_prog.search(see_data)
                if agent_match is not None:
                    obs.append(("loc", [int(agent_match.group("row")), int(agent_match.group("col"))]))

                ball_prog = re.compile("\(b (?P<row>\w+) (?P<col>\w+)\)")
                ball_match = ball_prog.search(msg)
                if ball_match is not None:
                    obs.append(("ball", [int(ball_match.group("row")), int(ball_match.group("col"))]))

                player_prog = re.compile("\((?P<team>[lr]) (?P<unum>\w+) (?P<row>\w+) (?P<col>\w+)\)")
                player_iter = player_prog.finditer(msg)
                for player_match in player_iter:
                    team = player_match.group("team") == "l"
                    obs.append(("player", [team, int(player_match.group("unum")), int(player_match.group("row")), int(player_match.group("col"))]))

                score_prog = re.compile("\(score (?P<left_score>\w+) (?P<right_score>\w+)\)")
                score_match = score_prog.search(msg)
                if score_match is not None:
                    obs.append(("score", [int(score_match.group("left_score")), int(score_match.group("right_score"))]))

        return obs

    # Sets the agent's home position to the given row and column
    def set_home(self, row, col):
        self.home_row = row
        self.home_col = col
        self.sock.send("(home " + str(row) + " " + str(col) + ")")

    # Sends an action to the server valid actions are:
    # action: "move", action_data: int representing direction (0 - hold, 1 - east, 2 - south, 3 - west, 4 - north,
    # 5 - north-east, 6 - south-east, 7 - south-west, 8 - north-west)
    # action: "pass", action_data: <target_uni_number>
    # action: "restart" - restarts the game, action_data: boolean - increments own team's score before restarting
    # if true otherwise increments opponent's team's score
    def send_action(self, action, action_data=None):
        if action == "move":
            action_array = [
                "(hold)", "(move east)", "(move south)", "(move west)", "(move north)", "(move north-east)",
                "(move south-east)", "(move south-west)", "(move north-west)"
            ]
            self.sock.send(action_array[action_data])
        elif action == "pass":
            self.sock.send("(pass " + str(action_data) + ")")
        elif action == "restart":
            if action_data:
                self.sock.send("episode-timeout our-pass")
            else:
                self.sock.send("episode-timeout our-fail")