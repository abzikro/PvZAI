class Action:
    pass

class PlaceAttackerAction(Action):
    def __init__(self, attacker):
        self.attacker = attacker

class PlaceSunflowerAction(Action):
    def __init__(self, sunflower):
        self.sunflower = sunflower

class PlaceDefenderAction(Action):
    def __init__(self, defender):
        self.defender = defender

class PlaceBomberAction(Action):
    def __init__(self, bomber):
        self.bomber = bomber

class CollectSunAction(Action):
    def __init__(self, position):
        self.position = position

    def __str__(self):
        return f"Collect Sun at {self.position}"


class DoNothingAction(Action):
    def __str__(self):
        return "Do Nothing"

class DoNothingActionMain(Action):
    def __str__(self):
        return "Do Nothing"
