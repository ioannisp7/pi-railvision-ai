import unittest

from railway_elements import (
    RailwayElements,
    TrackSection,
    Turnout,
    TurnoutPosition
)


class TestRailwayElements(unittest.TestCase):

    def setUp(self):
        self.railway_elements = RailwayElements()

    # Track section tests

    def test_add_track_section(self):
        section = self.railway_elements.add_track_section("SECTION_A")

        self.assertIsInstance(section, TrackSection)
        self.assertEqual(section.name, "SECTION_A")
        self.assertFalse(section.power_enabled)

    def test_set_track_section_power_on(self):
        self.railway_elements.add_track_section("SECTION_A")

        self.railway_elements.set_track_section_power(
            "SECTION_A",
            True
        )

        section = self.railway_elements.get_track_section("SECTION_A")

        self.assertTrue(section.power_enabled)

    def test_set_track_section_power_off(self):
        self.railway_elements.add_track_section("SECTION_A")

        self.railway_elements.set_track_section_power(
            "SECTION_A",
            True
        )

        self.railway_elements.set_track_section_power(
            "SECTION_A",
            False
        )

        section = self.railway_elements.get_track_section("SECTION_A")

        self.assertFalse(section.power_enabled)

    def test_get_track_section(self):
        created_section = self.railway_elements.add_track_section("SECTION_A")

        retrieved_section = self.railway_elements.get_track_section("SECTION_A")

        self.assertIs(created_section, retrieved_section)

    def test_remove_track_section(self):
        self.railway_elements.add_track_section("SECTION_A")

        self.railway_elements.remove_track_section("SECTION_A")

        self.assertNotIn(
            "SECTION_A",
            self.railway_elements.track_sections
        )

    def test_add_duplicate_track_section(self):
        self.railway_elements.add_track_section("SECTION_A")

        with self.assertRaises(ValueError):
            self.railway_elements.add_track_section("SECTION_A")

    def test_get_nonexistent_track_section(self):
        with self.assertRaises(KeyError):
            self.railway_elements.get_track_section("SECTION_A")

    def test_invalid_track_section_power(self):
        self.railway_elements.add_track_section("SECTION_A")

        with self.assertRaises(TypeError):
            self.railway_elements.set_track_section_power(
                "SECTION_A",
                "ON"
            )

        section = self.railway_elements.get_track_section("SECTION_A")

        self.assertFalse(section.power_enabled)

    # Turnout tests

    def test_add_turnout(self):
        turnout = self.railway_elements.add_turnout("TURNOUT_A")

        self.assertIsInstance(turnout, Turnout)
        self.assertEqual(turnout.name, "TURNOUT_A")
        self.assertIsNone(turnout.position)

    def test_set_turnout_position_left(self):
        self.railway_elements.add_turnout("TURNOUT_A")

        self.railway_elements.set_turnout_position(
            "TURNOUT_A",
            TurnoutPosition.LEFT
        )

        turnout = self.railway_elements.get_turnout("TURNOUT_A")

        self.assertEqual(turnout.position, TurnoutPosition.LEFT)

    def test_set_turnout_position_right(self):
        self.railway_elements.add_turnout("TURNOUT_A")

        self.railway_elements.set_turnout_position(
            "TURNOUT_A",
            TurnoutPosition.RIGHT
        )

        turnout = self.railway_elements.get_turnout("TURNOUT_A")

        self.assertEqual(turnout.position, TurnoutPosition.RIGHT)

    def test_get_turnout(self):
        created_turnout = self.railway_elements.add_turnout("TURNOUT_A")

        retrieved_turnout = self.railway_elements.get_turnout("TURNOUT_A")

        self.assertIs(created_turnout, retrieved_turnout)

    def test_remove_turnout(self):
        self.railway_elements.add_turnout("TURNOUT_A")

        self.railway_elements.remove_turnout("TURNOUT_A")

        self.assertNotIn(
            "TURNOUT_A",
            self.railway_elements.turnouts
        )

    def test_add_duplicate_turnout(self):
        self.railway_elements.add_turnout("TURNOUT_A")

        with self.assertRaises(ValueError):
            self.railway_elements.add_turnout("TURNOUT_A")

    def test_get_nonexistent_turnout(self):
        with self.assertRaises(KeyError):
            self.railway_elements.get_turnout("TURNOUT_A")

    def test_invalid_turnout_position(self):
        self.railway_elements.add_turnout("TURNOUT_A")

        with self.assertRaises(TypeError):
            self.railway_elements.set_turnout_position(
                "TURNOUT_A",
                "LEFT"
            )

        turnout = self.railway_elements.get_turnout("TURNOUT_A")

        self.assertIsNone(turnout.position)


if __name__ == "__main__":
    unittest.main()