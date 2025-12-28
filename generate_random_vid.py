from animations import AnimationConfig, render_animation
import random
from itertools import count
import datetime
from manim import RandomColorGenerator, random_bright_color
from pathlib import Path
import math

# modded random.uniform to get only two decimal places
def random_uniform_2dec(a, b):
    return round(random.uniform(a, b), 2)

def get_random_point_in_circle(radius):
    angle = random_uniform_2dec(0, 2 * math.pi)
    r = radius * (random_uniform_2dec(0, 1) ** 0.5)
    x = round(r * math.cos(angle), 2)
    y = round(r * math.sin(angle), 2)
    return [x, y, 0]

def main( generation_num: int ) -> None:
    _SEED = datetime.datetime.now().timestamp()
    
    random.seed(_SEED)
    rnd_color = RandomColorGenerator(seed=_SEED)
    
    SOUND_EFFECTS = list(Path("sound_effect").glob("*.wav"))
    ANIMATION_TYPE = random.choices(
        ["BouncingDot", "BouncingDots"],
        weights=[0.25, 0.75]
    )[0]

    # Create random config for each value in app.cfg
    config_dict = {
        "CIRCLE_RADIUS": random_uniform_2dec(2.5, 4.0),
        "CIRCLE_COLOR": random_bright_color(),
        "CIRCLE_STROKE_WIDTH": random_uniform_2dec(1.5, 2.5),
        
        "ENABLE_TRAIL": random.choices(
            [True, False],
            weights=[0.7, 0.3]
        )[0],
        "TRAIL_WIDTH": random_uniform_2dec(1.0, 3.0),
        "TRAIL_FADING_TIME": random.choice(
            [
                None,                         # no fading
                random_uniform_2dec(0.3, 1.0),   # short fade
                random_uniform_2dec(1.0, 3.0),   # medium fade
                random_uniform_2dec(5.0, 10.0),  # long fade (almost permanent)
            ]
        ),
        "TRAIL_OPACITY": random_uniform_2dec(0.4, 1.0),
        
        # need only the name
        "SOUND_EFFECT": str(random.choice(SOUND_EFFECTS).name),
    }

    if ANIMATION_TYPE == "BouncingDot":
        dot_pos = get_random_point_in_circle(config_dict["CIRCLE_RADIUS"] - 1.0)
        MAX_SPEED = random_uniform_2dec(5.0, 15.0)
        config_dict.update({
            "DOT_COLOR": rnd_color.next(),
            "DOT_RADIUS": random_uniform_2dec(0.15, 0.4),
            "DOT_START_X": dot_pos[0],
            "DOT_START_Y": dot_pos[1],
            "INITIAL_VELOCITY_X": random_uniform_2dec(-MAX_SPEED, MAX_SPEED),
            "INITIAL_VELOCITY_Y": random_uniform_2dec(-MAX_SPEED, MAX_SPEED),
            "DAMPING": random_uniform_2dec(0.85, 0.95),
            
            "TRAIL_COLOR": rnd_color.next(),
        })
    else:
        MAX_SPEED = random_uniform_2dec(5.0, 20.0)
        config_dict.update({
            # Update Gravity for more variety
            "GRAVITY_X": random.choices(
                [0.0, random_uniform_2dec(-2.0, 2.0)],
                weights=[0.25, 0.75]
            )[0],
            "GRAVITY_Y": random.choices(
                [-9.8, random_uniform_2dec(-6.0, -4.0), random_uniform_2dec(-12.0, -15.0)],
                weights=[0.25, 0.375, 0.375]
            )[0],
        })
        number_of_dots = random.randint(2, 7)
        DOTS_JSON = []
        for _ in range(number_of_dots):
            dot_pos = get_random_point_in_circle(config_dict["CIRCLE_RADIUS"] - 1.0)
            dot_info = {
                "color": rnd_color.next(),
                "radius": random_uniform_2dec(0.15, 0.35),
                "start_pos": dot_pos,
                "initial_velocity": [
                    random_uniform_2dec(-MAX_SPEED, MAX_SPEED),
                    random_uniform_2dec(-MAX_SPEED, MAX_SPEED),
                    0
                ],
                "damping": random_uniform_2dec(0.95, 0.995),
            }
            DOTS_JSON.append(dot_info)
        config_dict["DOTS_JSON"] = DOTS_JSON

    i = generation_num
    config = AnimationConfig()
    config.override(config_dict)

    output = render_animation(
            animation_name=ANIMATION_TYPE,
            config=config,
            output_name=f"Procedural Zen #{i}"
    )
        
    print(f"Animation saved to: {output}")
        