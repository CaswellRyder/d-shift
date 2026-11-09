"""
lost_procedure.py - what to do when no target is found.

    yaw_deg, elev_change_m, travel_m = lost_procedure()

Returns the same three values, in the same order and units, as
Target_vector_creation.target_vector():
    yaw_deg        degrees to turn (+ right, - left)
    elev_change_m  meters to climb (+) or descend (-)
    travel_m       meters to travel forward

For now: turn 120 degrees left in place, no climb, no travel, then look again.
"""

# ================= SETTINGS =================
SEARCH_YAW_DEG   = -120.0   # negative = left
SEARCH_ELEV_M    = 0.0
SEARCH_TRAVEL_M  = 0.0
# ============================================


def lost_procedure():
    return SEARCH_YAW_DEG, SEARCH_ELEV_M, SEARCH_TRAVEL_M


if __name__ == "__main__":
    print(lost_procedure())