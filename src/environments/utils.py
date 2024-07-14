
def get_landing_zone():
    # This would be a fixed or randomly generated position for the landing zone
    return (50, 0)  # Example coordinates for the landing zone

def get_angle(action):
    # Calculate the new angle of the lander based on the action
    # For simplicity, let's assume actions change the angle by a fixed amount
    angle_change = action.get('angle_change', 0)
    return angle_change

def get_velocity(action):
    # Calculate the new velocity of the lander based on the action
    vel_x_change = action.get('vel_x_change', 0)
    vel_y_change = action.get('vel_y_change', 0)
    return (vel_x_change, vel_y_change)

def get_position(action):
    # Calculate the new position of the lander based on the action
    pos_x_change = action.get('pos_x_change', 0)
    pos_y_change = action.get('pos_y_change', 0)
    return (pos_x_change, pos_y_change)

def get_fuel(action):
    # Calculate the remaining fuel based on the action
    fuel_change = action.get('fuel_change', 0)
    return fuel_change
