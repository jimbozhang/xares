from typing import Any, Dict
import torch
from xares.task_base import TaskBase, TaskConfig


class FSD50kTask(TaskBase):
    def __init__(self, encoder):
        data_key = "soundevent"
        self.class_label_maps = {
            "Accelerating_and_revving_and_vroom": 0,
            "Accordion": 1,
            "Acoustic_guitar": 2,
            "Aircraft": 3,
            "Alarm": 4,
            "Animal": 5,
            "Applause": 6,
            "Bark": 7,
            "Bass_drum": 8,
            "Bass_guitar": 9,
            "Bathtub_(filling_or_washing)": 10,
            "Bell": 11,
            "Bicycle": 12,
            "Bicycle_bell": 13,
            "Bird": 14,
            "Bird_vocalization_and_bird_call_and_bird_song": 15,
            "Boat_and_Water_vehicle": 16,
            "Boiling": 17,
            "Boom": 18,
            "Bowed_string_instrument": 19,
            "Brass_instrument": 20,
            "Breathing": 21,
            "Burping_and_eructation": 22,
            "Bus": 23,
            "Buzz": 24,
            "Camera": 25,
            "Car": 26,
            "Car_passing_by": 27,
            "Cat": 28,
            "Chatter": 29,
            "Cheering": 30,
            "Chewing_and_mastication": 31,
            "Chicken_and_rooster": 32,
            "Child_speech_and_kid_speaking": 33,
            "Chime": 34,
            "Chink_and_clink": 35,
            "Chirp_and_tweet": 36,
            "Chuckle_and_chortle": 37,
            "Church_bell": 38,
            "Clapping": 39,
            "Clock": 40,
            "Coin_(dropping)": 41,
            "Computer_keyboard": 42,
            "Conversation": 43,
            "Cough": 44,
            "Cowbell": 45,
            "Crack": 46,
            "Crackle": 47,
            "Crash_cymbal": 48,
            "Cricket": 49,
            "Crow": 50,
            "Crowd": 51,
            "Crumpling_and_crinkling": 52,
            "Crushing": 53,
            "Crying_and_sobbing": 54,
            "Cupboard_open_or_close": 55,
            "Cutlery_and_silverware": 56,
            "Cymbal": 57,
            "Dishes_and_pots_and_pans": 58,
            "Dog": 59,
            "Domestic_animals_and_pets": 60,
            "Domestic_sounds_and_home_sounds": 61,
            "Door": 62,
            "Doorbell": 63,
            "Drawer_open_or_close": 64,
            "Drill": 65,
            "Drip": 66,
            "Drum": 67,
            "Drum_kit": 68,
            "Electric_guitar": 69,
            "Engine": 70,
            "Engine_starting": 71,
            "Explosion": 72,
            "Fart": 73,
            "Female_singing": 74,
            "Female_speech_and_woman_speaking": 75,
            "Fill_(with_liquid)": 76,
            "Finger_snapping": 77,
            "Fire": 78,
            "Fireworks": 79,
            "Fixed-wing_aircraft_and_airplane": 80,
            "Fowl": 81,
            "Frog": 82,
            "Frying_(food)": 83,
            "Gasp": 84,
            "Giggle": 85,
            "Glass": 86,
            "Glockenspiel": 87,
            "Gong": 88,
            "Growling": 89,
            "Guitar": 90,
            "Gull_and_seagull": 91,
            "Gunshot_and_gunfire": 92,
            "Gurgling": 93,
            "Hammer": 94,
            "Hands": 95,
            "Harmonica": 96,
            "Harp": 97,
            "Hi-hat": 98,
            "Hiss": 99,
            "Human_group_actions": 100,
            "Human_voice": 101,
            "Idling": 102,
            "Insect": 103,
            "Keyboard_(musical)": 104,
            "Keys_jangling": 105,
            "Knock": 106,
            "Laughter": 107,
            "Liquid": 108,
            "Livestock_and_farm_animals_and_working_animals": 109,
            "Male_singing": 110,
            "Male_speech_and_man_speaking": 111,
            "Mallet_percussion": 112,
            "Marimba_and_xylophone": 113,
            "Mechanical_fan": 114,
            "Mechanisms": 115,
            "Meow": 116,
            "Microwave_oven": 117,
            "Motor_vehicle_(road)": 118,
            "Motorcycle": 119,
            "Music": 120,
            "Musical_instrument": 121,
            "Ocean": 122,
            "Organ": 123,
            "Packing_tape_and_duct_tape": 124,
            "Percussion": 125,
            "Piano": 126,
            "Plucked_string_instrument": 127,
            "Pour": 128,
            "Power_tool": 129,
            "Printer": 130,
            "Purr": 131,
            "Race_car_and_auto_racing": 132,
            "Rail_transport": 133,
            "Rain": 134,
            "Raindrop": 135,
            "Ratchet_and_pawl": 136,
            "Rattle": 137,
            "Rattle_(instrument)": 138,
            "Respiratory_sounds": 139,
            "Ringtone": 140,
            "Run": 141,
            "Sawing": 142,
            "Scissors": 143,
            "Scratching_(performance_technique)": 144,
            "Screaming": 145,
            "Screech": 146,
            "Shatter": 147,
            "Shout": 148,
            "Sigh": 149,
            "Singing": 150,
            "Sink_(filling_or_washing)": 151,
            "Siren": 152,
            "Skateboard": 153,
            "Slam": 154,
            "Sliding_door": 155,
            "Snare_drum": 156,
            "Sneeze": 157,
            "Speech": 158,
            "Speech_synthesizer": 159,
            "Splash_and_splatter": 160,
            "Squeak": 161,
            "Stream": 162,
            "Strum": 163,
            "Subway_and_metro_and_underground": 164,
            "Tabla": 165,
            "Tambourine": 166,
            "Tap": 167,
            "Tearing": 168,
            "Telephone": 169,
            "Thump_and_thud": 170,
            "Thunder": 171,
            "Thunderstorm": 172,
            "Tick": 173,
            "Tick-tock": 174,
            "Toilet_flush": 175,
            "Tools": 176,
            "Traffic_noise_and_roadway_noise": 177,
            "Train": 178,
            "Trickle_and_dribble": 179,
            "Truck": 180,
            "Trumpet": 181,
            "Typewriter": 182,
            "Typing": 183,
            "Vehicle": 184,
            "Vehicle_horn_and_car_horn_and_honking": 185,
            "Walk_and_footsteps": 186,
            "Water": 187,
            "Water_tap_and_faucet": 188,
            "Waves_and_surf": 189,
            "Whispering": 190,
            "Whoosh_and_swoosh_and_swish": 191,
            "Wild_animals": 192,
            "Wind": 193,
            "Wind_chime": 194,
            "Wind_instrument_and_woodwind_instrument": 195,
            "Wood": 196,
            "Writing": 197,
            "Yell": 198,
            "Zipper_(clothing)": 199,
        }

        def manyhot_label_map(x: Dict[str, Any]):
            target_ids = [self.class_label_maps[item] for item in x[data_key].split(";")]
            target_vec = torch.zeros(len(self.class_label_maps))
            target_vec.scatter_(0, torch.tensor(target_ids), 1)
            return target_vec.tolist()

        task_config = TaskConfig(
            batch_size_train=64,
            train_split="fsdk50_train",
            test_split="fsdk50_eval",
            valid_split="fsdk50_validation",
            criterion="BCEWithLogitsLoss",
            metric='mAP',
            zenodo_id="TODO",
            output_dim=len(self.class_label_maps),
            label_processor=manyhot_label_map,
        )
        super().__init__(encoder, config=task_config)

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
