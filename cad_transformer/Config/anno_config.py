'''config file'''

# color generated from: https://mdigi.tools/color-shades/#c8d3e5
super_class_dict = {"door":{"ids":[1,2,3,4,5,6]}, \
                    "window":{"ids":[7,8,9,10]}, \
                    "stairs":{"ids":[28]}, \
                    "home_appliance":{"ids":[18, 20, 24]}, \
                    "furniture":{"ids":[11,12,13,14,15,16,17,19,21,22,23,25,26,27]}, \
                    "equipment":{"ids":[29,30]}, \
                    "countable":{"ids":[_ for _ in range(1, 31)]}, \
                    "uncountable":{"ids":[31,32,33,34,35]}}

color_pallete = {
    0: (0, 0, 0),
    1: (241, 226, 226),
    2: (214, 168, 168),
    3: (187, 111, 111),  #door
    4: (187, 111, 111),
    5: (87, 41, 41),
    6: (29, 14, 14),  #door
    7: (226, 232, 241),
    8: (168, 186, 214),
    9: (111, 140, 187),
    10: (68, 97, 144),  #window
    28: (244, 247, 50),  #stairs
    18: (247, 214, 253),
    20: (231, 132, 250),
    24: (103, 5, 123),  #home appliance
    11: (245, 250, 245),
    12: (224, 241, 225),
    13: (203, 232, 204),
    14: (182, 223, 184),  #furniture
    15: (161, 214, 164),
    16: (140, 205, 143),
    17: (119, 196, 123),
    19: (98, 187, 103),  #furniture
    21: (78, 177, 83),
    22: (68, 157, 73),
    23: (59, 136, 63),
    25: (50, 115, 53),
    26: (41, 94, 44),
    27: (32, 73, 34),  #furniture
    29: (239, 220, 220),
    30: (177, 78, 78),  #equipment
    31: (222, 210, 179),
    32: (200, 180, 128),
    33: (177, 150, 78),
    34: (127, 107, 55),
    35: (76, 64, 33),  #stuff
}

super_tiny = 10
tiny = 15.
small = 20.
middle = 30.
large = 50.
super_large = 80.
bandwidth_dict = {
    1: small,
    2: middle,
    3: middle,  #door
    4: middle,
    5: middle,
    6: middle,  #door
    7: small,
    8: small,
    9: small,
    10: small,  #window
    28: super_large,  #stairs
    18: small,
    20: middle,
    24: middle,  #home appliance
    11: middle,
    12: super_large,
    13: tiny,
    14: middle,  #furniture
    15: small,
    16: small,
    17: tiny,
    19: tiny,  #furniture
    21: middle,
    22: middle,
    23: middle,
    25: super_tiny,
    26: super_tiny,
    27: small,  #furniture
    29: super_large,
    30: super_large * 4,  #equipment
}

TRAIN_CLASS_MAP_DICT = {
    'all': {str(i): [i]
            for i in range(36)},
    'wall': {
        '1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 33, 34],
        'others': 0,
    },
}


class AnnoList:

    def __init__(self, train_mode='all'):
        self.train_mode = train_mode

        self.updateLabel()

        self.anno_list_all_reverse = {
            v: k
            for k, v in self.anno_list_all.items()
        }
        return

    def setAll(self):
        self.anno_list_all = {
            'BG': 0,
            'single door': 1,
            'double door': 2,
            'sliding door': 3,
            'folding door': 4,
            'revolving door': 5,
            'rolling door': 6,
            'window': 7,
            'bay window': 8,
            'blind window': 9,
            'opening symbol': 10,
            'sofa': 11,
            'bed': 12,
            'chair': 13,
            'table': 14,
            'TV cabinet': 15,
            'Wardrobe': 16,
            'cabinet': 17,
            'gas stove': 18,
            'sink': 19,
            'refrigerator': 20,
            'airconditioner': 21,
            'bath': 22,
            'bath tub': 23,
            'washing machine': 24,  #TODO
            'squat toilet': 25,
            'urinal': 26,
            'toilet': 27,
            'stairs': 28,
            'elevator': 29,
            'escalator': 30,
            'row chairs': 31,
            'parking spot': 32,
            'wall': 33,
            'curtain wall': 34,
            'railing': 35,
        }
        return True

    def setWall(self):
        self.anno_list_all = {
            'BG': 0,
            'wall': 1,
        }

    def updateLabel(self):
        assert self.train_mode in TRAIN_CLASS_MAP_DICT

        if self.train_mode == 'all':
            return self.setAll()
        if self.train_mode == 'wall':
            return self.setWall()
        print("[ERROR][AnnoList::updateLabel]")
        print("\t train_mode [" + self.train_mode + "] has not been defined!")
        return False
