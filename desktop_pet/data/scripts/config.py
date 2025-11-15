# Configuration file for the application

from glob import glob

class Config(object):
    switch_action_dict = {
        '眨眼睛': (r'data\pets\eye\*.gif', 40),
        '腹黑': (r'data\pets\flft_lf\*.gif', 45),
        '画画': (r'data\pets\glgl\*.gif', 40),
        '举手': (r'data\pets\iwrt\*.gif', 40),
        '思考': (r'data\pets\lnft\*.gif', 45),
        '认真思考': (r'data\pets\lnft_ll\*.gif', 45),
        '摇摆': (r'data\pets\loop\*.gif', 30),
        '惊讶': (r'data\pets\nyya\*.gif', 40),
        '递贴纸': (r'data\pets\send_trkk\*.gif', 40),
    }

    @staticmethod
    def get_pictures(path_template):
        pics = glob(path_template)
        return pics[:]


















