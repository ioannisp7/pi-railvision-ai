from dataclasses import dataclass
from enum import Enum


class TurnoutPosition(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass
class TrackSection:
    name: str
    power_enabled: bool = False


@dataclass
class Turnout:
    name: str
    position: TurnoutPosition | None = None


class RailwayElements:

    def __init__(self):
        self.track_sections = {}
        self.turnouts = {}

    def validate_name(self, name):
        if not isinstance(name, str):
            raise TypeError("Element name must be a string.")

        name = name.strip()

        if not name:
            raise ValueError("Element name cannot be empty.")

        return name

    # Track sections

    def add_track_section(self, name):
        name = self.validate_name(name)

        if name in self.track_sections:
            raise ValueError(
                f"Track section '{name}' already exists."
            )

        track_section = TrackSection(name)

        self.track_sections[name] = track_section

        return track_section

    def get_track_section(self, name):
        name = self.validate_name(name)

        if name not in self.track_sections:
            raise KeyError(
                f"Track section '{name}' does not exist."
            )

        return self.track_sections[name]

    def remove_track_section(self, name):
        track_section = self.get_track_section(name)

        del self.track_sections[track_section.name]

    def set_track_section_power(self, name, power_enabled):
        if not isinstance(power_enabled, bool):
            raise TypeError("Power state must be True or False.")

        track_section = self.get_track_section(name)

        track_section.power_enabled = power_enabled

    # Turnouts

    def add_turnout(self, name):
        name = self.validate_name(name)

        if name in self.turnouts:
            raise ValueError(
                f"Turnout '{name}' already exists."
            )

        turnout = Turnout(name)

        self.turnouts[name] = turnout

        return turnout

    def get_turnout(self, name):
        name = self.validate_name(name)

        if name not in self.turnouts:
            raise KeyError(
                f"Turnout '{name}' does not exist."
            )

        return self.turnouts[name]

    def remove_turnout(self, name):
        turnout = self.get_turnout(name)

        del self.turnouts[turnout.name]

    def set_turnout_position(self, name, position):
        if not isinstance(position, TurnoutPosition):
            raise TypeError(
                "Position must be a TurnoutPosition value."
            )

        turnout = self.get_turnout(name)

        turnout.position = position